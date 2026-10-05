"""Release-frame-only offline metrics for task 87 checkpoints (plain vs release4x).

"Release frames" = exactly the samples the --release-weight run upweighted:
frames t whose 30-step action chunk contains a gripper closed -> open switch
(frames r-29..r-1 before each release r, per arm; train.release_mask). Control
frames = r-90..r-31 (gripper still closed, recorded chunk has no release) --
an over-trained release head would open early there.

--mode loss    teacher-forced training loss (compute_detailed_loss, train=True)
               on ALL release-window samples and on the control samples of the
               eval episodes; per-sample mean of action loss and of the
               releasing arm's gripper loss.
--mode timing  full policy (flow sampling) on every 2nd release-window frame
               and every 10th control frame: when does the predicted chunk
               open the releasing gripper vs the recording (frames), missed
               releases, early/false openings, gripper and action MAE.

One mode per process (a second LeRobotDataset construction in one process
hits the `datasets` bug, see train_one_task.py). Prints `EVAL_RELEASE_JSON:{...}`.
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
import numpy as np  # noqa: E402

from b1k.training import config as _config  # noqa: E402
from b1k.training import data_loader as _dl  # noqa: E402
from train_one_task import ILIA_ASSETS_BASE_DIR  # noqa: E402

ARM_DIMS = {"left": 14, "right": 22}
H = 30
WINDOW = range(1, H)          # t = r - k, k = 1..29 (chunk contains the switch)
CONTROL = range(31, 91)       # t = r - k, k = 31..90 (still holding, no switch in chunk)


def find_events(raw):
    """Release events (dataset index of r, arm) from the recorded per-frame actions."""
    hf = raw.hf_dataset.select_columns(["episode_index", "frame_index", "action"]).with_format("numpy")
    ep = np.asarray(hf["episode_index"]).astype(int)
    act = np.stack(hf["action"]).astype(np.float32)
    events = []
    for arm, d in ARM_DIMS.items():
        x = act[:, d]
        sw = np.where((x[:-1] < 0) & (x[1:] > 0) & (ep[:-1] == ep[1:]))[0] + 1
        events += [(int(r), arm) for r in sw]
    return sorted(events), ep, act


def frames_for(events, ep, ks):
    out = []
    for r, arm in events:
        for k in ks:
            t = r - k
            if t >= 0 and ep[t] == ep[r]:
                out.append((t, r, arm))
    return out


def make_cfg(episodes, batch_size=16, num_workers=16, params=None):
    from b1k.training import weight_loaders
    cfg = _config.get_config("pi_behavior_b1k_fast_100task")
    return dataclasses.replace(
        cfg, exp_name="eval_tmp", assets_base_dir=ILIA_ASSETS_BASE_DIR, batch_size=batch_size,
        num_workers=num_workers, wandb_enabled=False, seed=0,
        **({"weight_loader": weight_loaders.PiBehaviorWeightLoader(params)} if params else {}),
        model=dataclasses.replace(cfg.model, use_correlated_noise=True),
        data=dataclasses.replace(cfg.data, base_config=dataclasses.replace(cfg.data.base_config, episodes_index=episodes)),
    )


def mode_loss(ckpt: Path, episodes: list[int]):
    import jax
    import jax.numpy as jnp
    import torch
    from flax import nnx
    from openpi.training import data_loader as _odl
    from openpi.training import sharding
    import train as _train

    cfg = make_cfg(episodes, params=str(ckpt / "params"))
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    raw = _dl.create_behavior_dataset(dc, action_horizon=cfg.model.action_horizon, seed=0)
    events, ep, _ = find_events(raw)
    groups = {"release_window": frames_for(events, ep, WINDOW), "control": frames_for(events, ep, CONTROL)}
    ds = _dl.transform_dataset(raw, dc)
    mesh = sharding.make_mesh(cfg.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    state, _ = _train.init_train_state(cfg, jax.random.key(0), mesh, resume=False, norm_stats=dc.norm_stats)
    graphdef, params = nnx.split(nnx.merge(state.model_def, state.params))
    del state

    @jax.jit
    def per_sample(params, rng, obs, actions):
        m = nnx.merge(graphdef, params)
        d = m.compute_detailed_loss(rng, obs, actions, train=True, num_flow_samples=cfg.num_flow_samples)
        return d["action_loss"], d["action_loss_left_gripper"], d["action_loss_right_gripper"], _train.release_mask(actions)

    res = dict(mode="loss", n_events=len(events))
    for name, frames in groups.items():
        idx = [t for t, _, _ in frames]
        arms = [a for _, _, a in frames]
        n = len(idx)
        pad = (-n) % cfg.batch_size
        sub = torch.utils.data.Subset(ds, idx + idx[:pad])
        loader = _dl.DataLoaderImpl(dc, _odl.TorchDataLoader(sub, local_batch_size=cfg.batch_size, sharding=data_sharding,
                                                             shuffle=False, num_batches=(n + pad) // cfg.batch_size,
                                                             num_workers=cfg.num_workers, seed=0))
        al, gl, gr, rm = [], [], [], []
        for b, (obs, actions) in enumerate(loader):
            with sharding.set_mesh(mesh):
                out = jax.device_get(per_sample(params, jax.random.fold_in(jax.random.key(1234), b), obs, actions))
            al.append(out[0]); gl.append(out[1]); gr.append(out[2]); rm.append(out[3])
        al, gl, gr, rm = (np.concatenate(v)[:n] for v in (al, gl, gr, rm))
        grip = np.where(np.array(arms) == "left", gl, gr)
        res[name] = dict(n=n, action_loss=float(al.mean()), releasing_gripper_loss=float(grip.mean()),
                         release_mask_rate=float(rm.mean()))
        print(f"{name}: n={n} action_loss={al.mean():.4f} gripper_loss={grip.mean():.4f} mask_rate={rm.mean():.3f}", flush=True)
    print("EVAL_RELEASE_JSON:" + json.dumps(res), flush=True)


def mode_timing(ckpt: Path, episodes: list[int]):
    import openpi.transforms as _transforms
    from b1k.models.pi_behavior_config import TASK_NUM_STAGES
    from b1k.policies import policy_config
    from b1k.shared import normalize as _normalize

    cfg = make_cfg(episodes)
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, use_correlated_noise=False))
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    raw = _dl.create_behavior_dataset(dc, action_horizon=cfg.model.action_horizon, seed=0)
    events, ep, act = find_events(raw)
    norm_dir = next(ckpt.glob("assets/**/norm_stats.json")).parent
    repack = _transforms.Group(inputs=[_transforms.RepackTransform({
        "observation/egocentric_camera": "observation.rgb.zed_link_camera_0",
        "observation/wrist_image_left": "observation.rgb.left_realsense_link_camera_0",
        "observation/wrist_image_right": "observation.rgb.right_realsense_link_camera_0",
        "observation/state": "observation.state",
        "task_index": "task_index", "timestamp": "timestamp", "episode_index": "episode_index", "index": "index",
        "tokenized_prompt": "tokenized_prompt", "tokenized_prompt_mask": "tokenized_prompt_mask", "subtask_state": "subtask_state",
    })])
    policy = policy_config.create_trained_policy(cfg, ckpt, repack_transforms=repack, norm_stats=_normalize.load(str(norm_dir)))
    ns = TASK_NUM_STAGES[87]
    ep_len = {e: int((ep == e).sum()) for e in set(ep.tolist())}

    def infer(t):
        s = raw[t]
        obs = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in s.items()}
        e = int(s["episode_index"])
        gt = max(0, min(int(float(s["timestamp"]) * 30.0 / (ep_len[e] / ns)), ns - 1))
        obs["tokenized_prompt"] = np.array([87, gt], dtype=np.int32)
        obs["tokenized_prompt_mask"] = np.array([True, True], dtype=bool)
        obs["subtask_state"] = np.array(gt, dtype=np.int32)
        return np.asarray(policy.infer(obs)["actions"])[:H, :23]

    win = frames_for(events, ep, range(1, H, 2))
    ctl = frames_for(events, ep, range(31, 91, 10))
    terr, missed, already_open, gmae, amae = [], 0, 0, [], []
    for n, (t, r, arm) in enumerate(win):
        d = ARM_DIMS[arm]
        p = infer(t)
        g = act[t:t + H]
        k_gt = r - t
        x = p[:, d]
        if x[0] > 0:
            already_open += 1
            terr.append(0 - k_gt)
        else:
            sw = np.where((x[:-1] <= 0) & (x[1:] > 0))[0]
            if len(sw):
                terr.append(int(sw[0] + 1) - k_gt)
            else:
                missed += 1
        h = min(len(g), len(p))
        gmae.append(float(np.abs(p[:h, d] - g[:h, d]).mean()))
        amae.append(float(np.abs(p[:h] - g[:h]).mean()))
        if n % 50 == 0:
            print(f"window {n}/{len(win)}", flush=True)
    false_open, cg = 0, []
    for t, r, arm in ctl:
        d = ARM_DIMS[arm]
        p = infer(t)
        g = act[t:t + H]
        if (p[:, d] > 0).any():
            false_open += 1
        h = min(len(g), len(p))
        cg.append(float(np.abs(p[:h, d] - g[:h, d]).mean()))
    terr = np.array(terr)
    res = dict(
        mode="timing", n_events=len(events), n_window_frames=len(win), n_control_frames=len(ctl),
        timing_mean_abs_err_frames=float(np.abs(terr).mean()) if len(terr) else None,
        timing_mean_signed_err_frames=float(terr.mean()) if len(terr) else None,
        timing_within_3_frames=float((np.abs(terr) <= 3).mean()) if len(terr) else None,
        missed_release_rate=missed / len(win), already_open_rate=already_open / len(win),
        window_gripper_mae=float(np.mean(gmae)), window_action_mae=float(np.mean(amae)),
        control_false_open_rate=false_open / len(ctl), control_gripper_mae=float(np.mean(cg)),
    )
    print("EVAL_RELEASE_JSON:" + json.dumps(res), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--mode", choices=["loss", "timing"], required=True)
    p.add_argument("--episodes", type=int, nargs="+", default=list(range(17400, 17406)))
    a = p.parse_args()
    (mode_loss if a.mode == "loss" else mode_timing)(Path(a.checkpoint_dir), a.episodes)
