"""Teacher-forced action error + stage accuracy of one checkpoint on one
episode (subprocess of eval_push.py; one episode per process, see
eval_loss_one.py for why). Adapted from the 2026-09-13 lag-tolerant harness
(tmp/task82_video_eval/eval_checkpoint.py) so numbers stay comparable with it:

- every `stride`-th frame, feed the recorded observation + the uniform-time
  ground-truth stage (frames split evenly over TASK_NUM_STAGES, as before),
  run the full policy (flow sampling, unnormalized output, no correction rules),
- compare the predicted action chunk with the recorded one (raw units):
  MAE of the first action, of the whole chunk, and per joint group,
- exact and within-one stage accuracy of the predicted stage.

Prints one line `EVAL_TF_JSON:{...}`.
"""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
import numpy as np  # noqa: E402

import openpi.transforms as _transforms  # noqa: E402
from b1k.models.pi_behavior_config import TASK_NUM_STAGES  # noqa: E402
from b1k.policies import policy_config  # noqa: E402
from b1k.shared import normalize as _normalize  # noqa: E402
from b1k.training import config as _config  # noqa: E402
from b1k.training.data_loader import create_behavior_dataset  # noqa: E402

# Raw 23-d action layout (same indices as compute_detailed_loss's per-joint losses).
GROUPS = {
    "base": slice(0, 3), "trunk": slice(3, 7), "left_arm": slice(7, 14), "left_gripper": slice(14, 15),
    "right_arm": slice(15, 22), "right_gripper": slice(22, 23),
}

p = argparse.ArgumentParser()
p.add_argument("--task-index", type=int, required=True)
p.add_argument("--episode-idx", type=int, required=True)
p.add_argument("--checkpoint-dir", required=True, help="step dir containing params/ and assets/")
p.add_argument("--stride", type=int, default=50)
args = p.parse_args()

ckpt = Path(args.checkpoint_dir)
# Checkpoints keep their own assets/; the rebuilt base has none -> Ilia's stats
# (what every checkpoint in this chain was trained with).
found = sorted(ckpt.glob("assets/**/norm_stats.json")) or sorted((REPO_ROOT / "outputs/assets_ilia").glob("**/norm_stats.json"))
norm_dir = found[0].parent
cfg = _config.get_config("pi_behavior_b1k_fast_100task")
cfg = dataclasses.replace(
    cfg,
    assets_base_dir=str(REPO_ROOT / "outputs/assets_ilia"),
    model=dataclasses.replace(cfg.model, use_correlated_noise=False),
    data=dataclasses.replace(cfg.data, base_config=dataclasses.replace(cfg.data.base_config, episodes_index=[args.episode_idx])),
)
data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
ds = create_behavior_dataset(data_config, action_horizon=cfg.model.action_horizon, seed=0)
repack = _transforms.Group(inputs=[_transforms.RepackTransform({
    "observation/egocentric_camera": "observation.rgb.zed_link_camera_0",
    "observation/wrist_image_left": "observation.rgb.left_realsense_link_camera_0",
    "observation/wrist_image_right": "observation.rgb.right_realsense_link_camera_0",
    "observation/state": "observation.state",
    "task_index": "task_index", "timestamp": "timestamp", "episode_index": "episode_index", "index": "index",
    "tokenized_prompt": "tokenized_prompt", "tokenized_prompt_mask": "tokenized_prompt_mask", "subtask_state": "subtask_state",
})])
policy = policy_config.create_trained_policy(cfg, ckpt, repack_transforms=repack, norm_stats=_normalize.load(str(norm_dir)))

ep_length = ds.meta.episodes[args.episode_idx].get("length")
num_stages = TASK_NUM_STAGES[args.task_index]


def conv(v):
    return v.numpy() if hasattr(v, "numpy") else v


gt_st, pr_st, err_first, err_chunk = [], [], [], []
t0 = time.time()
for i in range(0, len(ds), args.stride):
    s = ds[i]
    obs = {k: conv(v) for k, v in s.items()}
    gt = max(0, min(int(float(s["timestamp"]) * 30.0 / (ep_length / num_stages)), num_stages - 1))
    obs["tokenized_prompt"] = np.array([int(s["task_index"]), gt], dtype=np.int32)
    obs["tokenized_prompt_mask"] = np.array([True, True], dtype=bool)
    obs["subtask_state"] = np.array(gt, dtype=np.int32)
    r = policy.infer(obs)
    gt_a = np.asarray(conv(s["action"]))
    pr_a = np.asarray(r["actions"])
    h = min(len(gt_a), len(pr_a))
    err_first.append(np.abs(gt_a[0] - pr_a[0]))
    err_chunk.append(np.abs(gt_a[:h] - pr_a[:h]).mean(axis=0))
    gt_st.append(gt)
    pr_st.append(int(r["predicted_stage"]))
gt_st, pr_st = np.array(gt_st), np.array(pr_st)
ef, ec = np.stack(err_first), np.stack(err_chunk)
res = dict(
    task_index=args.task_index, episode_idx=args.episode_idx, stride=args.stride, frames=len(gt_st),
    exact_stage_accuracy=float((gt_st == pr_st).mean()),
    within_one_stage_accuracy=float((np.abs(gt_st - pr_st) <= 1).mean()),
    action_mae_first=float(ef.mean()), action_mae_chunk=float(ec.mean()),
    **{f"mae_chunk_{g}": float(ec[:, sl].mean()) for g, sl in GROUPS.items()},
    **{f"mae_first_{g}": float(ef[:, sl].mean()) for g, sl in GROUPS.items()},
    seconds=round(time.time() - t0),
)
print("EVAL_TF_JSON:" + json.dumps(res), flush=True)
