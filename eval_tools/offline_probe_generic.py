"""Generic offline probe (2026-10-06; generalised from offline_probe_task76.py).

Feeds recorded 2026 demo frames (3 cameras resize_with_pad 224, 61-d state, time-based stage label) to a checkpoint and
prints, per frame and over N noise samples, what the policy predicts for the 26 executed actions:
  grippers  : chunk min/max per hand and how many samples open (> 0) anywhere in the chunk, vs the demo
  base      : mean predicted base velocity (vx, vy, wz; robot frame) over the chunk vs the demo
The task id given to the model can differ from the episode's task (--prompt-task) to test how much the task embedding
changes the prediction (task-conditioning test).

  python offline_probe_generic.py --ckpt DIR --label NAME --chunk 087 --ep 17400 --task 87 [--prompt-task 76] \
      [--chain 1] --frames "release_R:1950,1977,1997;carry:1300,1450" [--samples 5] [--out res.json]
"""
import argparse
import dataclasses
import json
import os
import sys

import av
import numpy as np
import pyarrow.parquet as pq

REPO = "/workspace/persistent/repos/place_inside_4_tasks"
DEMO = "/home/kasm-user/demos2026"
EXEC = 26
L_GRIP, R_GRIP, BASE = 14, 22, slice(0, 3)
L_ARM, R_ARM = slice(7, 14), slice(15, 22)
CAMS = ("zed_link_camera_0", "left_realsense_link_camera_0", "right_realsense_link_camera_0")
sys.path.insert(0, os.path.join(REPO, "scripts"))


def episode_offset(chunk, ep, fps=30):
    m = pq.read_table(f"{DEMO}/meta/episodes/chunk-{chunk}/file-000.parquet").to_pandas()
    r = m[m.episode_index == ep].iloc[0]
    return int(round(float(r["videos/observation.rgb.zed_link_camera_0/from_timestamp"]) * fps))


def load_frames(cam, chunk, frames):
    path = f"{DEMO}/videos/observation.rgb.{cam}/chunk-{chunk}/file-000.mp4"
    want, out = set(frames), {}
    with av.open(path) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i in want:
                out[i] = fr.to_ndarray(format="rgb24")
            if i >= max(want):
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--label", required=True)
    ap.add_argument("--chunk", required=True); ap.add_argument("--ep", type=int, required=True)
    ap.add_argument("--task", type=int, required=True); ap.add_argument("--prompt-task", type=int)
    ap.add_argument("--chain", type=int, default=1, help="1 = 649-row 76-80-87 stage table, 0 = 627-row ext100")
    ap.add_argument("--frames", required=True); ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--config", default="pi_behavior_b1k_fast_100task"); ap.add_argument("--out")
    a = ap.parse_args()
    ptask = a.prompt_task if a.prompt_task is not None else a.task
    frames = {}
    for part in a.frames.split(";"):
        k, v = part.split(":")
        frames[k] = [int(x) for x in v.split(",")]

    from openpi_client.image_tools import resize_with_pad
    from b1k.training import config as _config
    from b1k.policies import policy_config as _policy_config
    from b1k.policies.b1k_policy import extract_state_from_proprio
    from b1k.models import pi_behavior as _pb, pi_behavior_config as _pbc
    if a.chain:
        _t = list(_pbc.TASK_NUM_STAGES); _t[80] = 10; _t[87] = 12; _t = tuple(_t)
        _pbc.TASK_NUM_STAGES = _pb.TASK_NUM_STAGES = _t
        _pb.TOTAL_TASK_STAGE_EMBEDDINGS = sum(_t)
        _pb.TASK_STAGE_OFFSETS = tuple([0] + [sum(_t[: i + 1]) for i in range(len(_t) - 1)])
    num_stages = _pb.TASK_NUM_STAGES[a.task]
    pnum_stages = _pb.TASK_NUM_STAGES[ptask]
    cfg = _config.get_config(a.config)
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, use_correlated_noise=True))
    policy = _policy_config.create_trained_policy(cfg, a.ckpt, sample_kwargs={"num_steps": 20})

    t = pq.read_table(f"{DEMO}/data/chunk-{a.chunk}/file-000.parquet").to_pandas()
    e = t[t.episode_index == a.ep].sort_values("frame_index")
    state61 = np.stack(e["observation.state"].values).astype(np.float32)
    act = np.stack(e["action"].values).astype(np.float32)
    n = len(e)
    off = episode_offset(a.chunk, a.ep)
    allf = sorted(f for fs in frames.values() for f in fs)
    imgs = {cam: load_frames(cam, a.chunk, [f + off for f in allf]) for cam in CAMS}

    rows = []
    for phase, fs in frames.items():
        for f in fs:
            st23 = extract_state_from_proprio(state61[f])
            gt = act[f:f + EXEC]
            stage = min(int(f / (n / num_stages)), num_stages - 1)
            pstage = min(int(stage * pnum_stages / num_stages), pnum_stages - 1)  # same relative progress
            obs = {
                "observation/egocentric_camera": resize_with_pad(imgs[CAMS[0]][f + off], 224, 224),
                "observation/wrist_image_left": resize_with_pad(imgs[CAMS[1]][f + off], 224, 224),
                "observation/wrist_image_right": resize_with_pad(imgs[CAMS[2]][f + off], 224, 224),
                "observation/state": state61[f],
                "tokenized_prompt": np.array([ptask, pstage], dtype=np.int32),
                "tokenized_prompt_mask": np.array([True, True]),
                "subtask_state": np.array(pstage, dtype=np.int32),
            }
            preds = [np.asarray(policy.infer(obs)["actions"])[:EXEC] for _ in range(a.samples)]
            r = {"label": a.label, "prompt_task": ptask, "phase": phase, "frame": f, "stage": pstage,
                 "base_pred": [p[:, BASE].mean(0).tolist() for p in preds], "base_gt": gt[:, BASE].mean(0).tolist(),
                 "state_grip": [float(st23[L_GRIP]), float(st23[R_GRIP])]}
            for side, arm in (("L", L_ARM), ("R", R_ARM)):
                dg = gt[-1, arm] - st23[arm]
                dps = [p[-1, arm] - st23[arm] for p in preds]
                r[f"arm_cos_{side}"] = float(np.median([dp @ dg / (np.linalg.norm(dp) * np.linalg.norm(dg) + 1e-8) for dp in dps]))
                r[f"arm_mag_{side}"] = float(np.median([np.linalg.norm(dp) / (np.linalg.norm(dg) + 1e-8) for dp in dps]))
                r[f"arm_gt_{side}"] = float(np.linalg.norm(dg))
            for side, g in (("L", L_GRIP), ("R", R_GRIP)):
                r[f"open_{side}"] = sum(1 for p in preds if p[:, g].max() > 0)
                r[f"pmin_{side}"] = float(np.median([p[:, g].min() for p in preds]))
                r[f"pmax_{side}"] = float(np.median([p[:, g].max() for p in preds]))
                r[f"gt_{side}"] = [float(gt[:, g].min()), float(gt[:, g].max())]
            rows.append(r)
            bp = np.mean(r["base_pred"], 0); bs = np.std(r["base_pred"], 0)
            print(f"{a.label:12s} task{ptask:<3d} {phase:12s} f{f:5d} st{pstage:2d} | grip state L {r['state_grip'][0]:+.2f} R {r['state_grip'][1]:+.2f} | "
                  f"PRED L {r['pmin_L']:+.2f}/{r['pmax_L']:+.2f} open {r['open_L']}/{a.samples}  R {r['pmin_R']:+.2f}/{r['pmax_R']:+.2f} open {r['open_R']}/{a.samples} | "
                  f"DEMO L {r['gt_L'][0]:+.2f}/{r['gt_L'][1]:+.2f} R {r['gt_R'][0]:+.2f}/{r['gt_R'][1]:+.2f} | "
                  f"arm cos/mag/demo-motion L {r['arm_cos_L']:+.2f}/{r['arm_mag_L']:.2f}/{r['arm_gt_L']:.2f} R {r['arm_cos_R']:+.2f}/{r['arm_mag_R']:.2f}/{r['arm_gt_R']:.2f} | "
                  f"base pred ({bp[0]:+.2f},{bp[1]:+.2f},{bp[2]:+.2f})±({bs[0]:.2f},{bs[1]:.2f},{bs[2]:.2f}) demo ({r['base_gt'][0]:+.2f},{r['base_gt'][1]:+.2f},{r['base_gt'][2]:+.2f})",
                  flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w")); print("saved", a.out)


if __name__ == "__main__":
    main()
