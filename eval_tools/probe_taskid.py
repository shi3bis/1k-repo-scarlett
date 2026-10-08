"""Task-ID sensitivity probe (2026-10-07). Does swapping the task id change the predicted actions more than sampling noise?

Feeds the SAME recorded demo frames (3 cameras, 61-d state) to a checkpoint with several task ids (stage = same relative
progress for each id), S noise samples per (frame, id), and compares the 26 executed actions:
  noise = mean |sample - mean of its id|           (sampling spread within one id)
  swap  = mean |mean(id A) - mean(id B)|            (difference caused by the task id), averaged over id pairs
  ratio = swap / noise   (~1 or below: the id is ignored; >> 1: the id changes the action)
Reported separately for base+torso+arms (dims 0-13, 15-21) and the grippers (14, 22).

  python probe_taskid.py --ckpt DIR --label NAME --chunk 000 --ep 0 --task 0 --ids 0,1,25 --frames 300,700,1050 \
      [--table base50|ext100|chain] [--samples 6]
"""
import argparse
import dataclasses
import itertools
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from offline_probe_generic import CAMS, DEMO, EXEC, REPO, episode_offset, load_frames  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "scripts"))
BODY = [i for i in range(23) if i not in (14, 22)]
GRIP = [14, 22]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--label", required=True)
    ap.add_argument("--chunk", required=True); ap.add_argument("--ep", type=int, required=True)
    ap.add_argument("--task", type=int, required=True); ap.add_argument("--ids", required=True)
    ap.add_argument("--frames", required=True); ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--table", choices=("base50", "ext100", "chain"), default="ext100"); ap.add_argument("--out")
    a = ap.parse_args()
    ids = [int(x) for x in a.ids.split(",")]
    frames = [int(x) for x in a.frames.split(",")]

    from openpi_client.image_tools import resize_with_pad
    from b1k.training import config as _config
    from b1k.policies import policy_config as _policy_config
    from b1k.models import pi_behavior as _pb, pi_behavior_config as _pbc
    cfg_name = "pi_behavior_b1k_fast_100task"
    if a.table == "chain":
        _t = list(_pbc.TASK_NUM_STAGES); _t[80] = 10; _t[87] = 12; _t = tuple(_t)
    elif a.table == "base50":
        _t = tuple(_pbc.TASK_NUM_STAGES[:50]); cfg_name = "pi_behavior_b1k_fast"
    else:
        _t = tuple(_pbc.TASK_NUM_STAGES)
    _pbc.TASK_NUM_STAGES = _pb.TASK_NUM_STAGES = _t
    _pb.TOTAL_TASK_STAGE_EMBEDDINGS = sum(_t)
    _pb.TASK_STAGE_OFFSETS = tuple([0] + [sum(_t[: i + 1]) for i in range(len(_t) - 1)])
    cfg = _config.get_config(cfg_name)
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, use_correlated_noise=True))
    policy = _policy_config.create_trained_policy(cfg, a.ckpt, sample_kwargs={"num_steps": 20})

    t = pq.read_table(f"{DEMO}/data/chunk-{a.chunk}/file-000.parquet").to_pandas()
    e = t[t.episode_index == a.ep].sort_values("frame_index")
    state61 = np.stack(e["observation.state"].values).astype(np.float32)
    n = len(e)
    off = episode_offset(a.chunk, a.ep)
    imgs = {cam: load_frames(cam, a.chunk, [f + off for f in frames]) for cam in CAMS}
    ns = _pb.TASK_NUM_STAGES[a.task]
    out = []
    print(f"{a.label}: task {a.task} ep {a.ep}, ids {ids}, {a.samples} samples; ratio = swap/noise (>>1 = task id matters)")
    for f in frames:
        stage = min(int(f / (n / ns)), ns - 1)
        preds = {}
        for tid in ids:
            pst = min(int(stage * _pb.TASK_NUM_STAGES[tid] / ns), _pb.TASK_NUM_STAGES[tid] - 1)
            obs = {
                "observation/egocentric_camera": resize_with_pad(imgs[CAMS[0]][f + off], 224, 224),
                "observation/wrist_image_left": resize_with_pad(imgs[CAMS[1]][f + off], 224, 224),
                "observation/wrist_image_right": resize_with_pad(imgs[CAMS[2]][f + off], 224, 224),
                "observation/state": state61[f],
                "tokenized_prompt": np.array([tid, pst], dtype=np.int32),
                "tokenized_prompt_mask": np.array([True, True]),
                "subtask_state": np.array(pst, dtype=np.int32),
            }
            preds[tid] = np.stack([np.asarray(policy.infer(obs)["actions"])[:EXEC] for _ in range(a.samples)])
        r = {"frame": f, "stage": stage}
        for name, dims in (("body", BODY), ("grip", GRIP)):
            noise = float(np.mean([np.abs(p[..., dims] - p[..., dims].mean(0)).mean() for p in preds.values()]))
            swap = float(np.mean([np.abs(preds[x][..., dims].mean(0) - preds[y][..., dims].mean(0)).mean()
                                  for x, y in itertools.combinations(ids, 2)]))
            r[name] = {"noise": noise, "swap": swap, "ratio": swap / max(noise, 1e-6)}
        out.append(r)
        print(f"  f{f:5d} st{stage:2d} | body noise {r['body']['noise']:.4f} swap {r['body']['swap']:.4f} ratio {r['body']['ratio']:5.2f}"
              f" | grip noise {r['grip']['noise']:.4f} swap {r['grip']['swap']:.4f} ratio {r['grip']['ratio']:5.2f}", flush=True)
    rb = np.median([r["body"]["ratio"] for r in out]); rg = np.median([r["grip"]["ratio"] for r in out])
    print(f"  MEDIAN ratio body {rb:.2f}  grip {rg:.2f}")
    if a.out:
        json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
