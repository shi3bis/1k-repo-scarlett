"""[task-76 variant of offline_probe_task35.py, 2026-10-04] Does the model predict RELEASING the grippers when it has
carried the trash can to the drop spot? 2026 demo episode 15200 (annotation 00760010): pick up trash_can 450-1020
(L closes 821, R closes 833), carry 1020-1560, place on floor 1560-2130 (R opens 1945, L opens 1957), then the glass.
Probe frames: grasp control (should close), carry (should stay closed), pre-release / release (should open), after.
Chain-76-80-87 649-row stage table; --noise corr is how these ckpts are served. Prints per frame the predicted
gripper chunk min/max (26 executed actions) vs the recorded chunk min/max; +1 open, -1 closed.
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
EP = 15200
TASK = 76
NUM_STAGES = 10
EXEC = 26
# Annotated skills in this episode (annotations/task-0055/episode_00550010.json).
FRAMES = {
    "grasp_ctrl(L821,R833)": [790, 801, 811],
    "carry": [1100, 1300, 1500],
    "place_early": [1600, 1750, 1850],
    "pre_release": [1900, 1915, 1925, 1935],
    "release(R1945,L1957)": [1945, 1955],
    "after_release": [1980, 2050],
}
FRAMES_ALIGNED = FRAMES
# 23-d action/state order (b1k_policy.extract_state_from_proprio): base vel 3, trunk 4, L arm 7, L grip 1, R arm 7, R grip 1
L_ARM, L_GRIP, R_ARM, R_GRIP = slice(7, 14), 14, slice(15, 22), 22

sys.path.insert(0, os.path.join(REPO, "scripts"))


def load_frames(cam, frames):
    path = f"{DEMO}/videos/observation.rgb.{cam}/chunk-076/file-000.mp4"
    want = set(frames)
    out = {}
    with av.open(path) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i in want:
                out[i] = fr.to_ndarray(format="rgb24")
            if i >= max(want):
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--config", default="pi_behavior_b1k_fast_100task")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--base50", action="store_true")
    ap.add_argument("--noise", choices=("corr", "indep"), default="corr")
    ap.add_argument("--frames", choices=("default", "aligned"), default="default")
    a = ap.parse_args()
    global FRAMES
    if a.frames == "aligned":
        FRAMES = FRAMES_ALIGNED

    from openpi_client.image_tools import resize_with_pad
    from b1k.training import config as _config
    from b1k.policies import policy_config as _policy_config
    from b1k.policies.b1k_policy import extract_state_from_proprio

    from b1k.models import pi_behavior as _pb, pi_behavior_config as _pbc
    _t = list(_pbc.TASK_NUM_STAGES); _t[80] = 10; _t[87] = 12; _t = tuple(_t)
    _pbc.TASK_NUM_STAGES = _pb.TASK_NUM_STAGES = _t
    _pb.TOTAL_TASK_STAGE_EMBEDDINGS = sum(_t)
    _pb.TASK_STAGE_OFFSETS = tuple([0] + [sum(_t[: i + 1]) for i in range(len(_t) - 1)])
    if a.base50:
        from b1k.models import pi_behavior as _pb, pi_behavior_config as _pbc
        _first50 = tuple(_pbc.TASK_NUM_STAGES[:50])
        _pb.TASK_NUM_STAGES = _first50
        _pb.TOTAL_TASK_STAGE_EMBEDDINGS = sum(_first50)
        _pb.TASK_STAGE_OFFSETS = tuple([0] + [sum(_first50[: i + 1]) for i in range(len(_first50) - 1)])
        a.config = "pi_behavior_b1k_fast"
    cfg = _config.get_config(a.config)
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, use_correlated_noise=(a.noise == "corr")))
    policy = _policy_config.create_trained_policy(cfg, a.ckpt, sample_kwargs={"num_steps": 20})

    t = pq.read_table(f"{DEMO}/data/chunk-076/file-000.parquet").to_pandas()
    e = t[t.episode_index == EP].sort_values("frame_index")
    state61 = np.stack(e["observation.state"].values).astype(np.float32)
    act = np.stack(e["action"].values).astype(np.float32)
    n = len(e)
    allf = sorted(f for fs in FRAMES.values() for f in fs)
    imgs = {cam: load_frames(cam, allf) for cam in ("zed_link_camera_0", "left_realsense_link_camera_0", "right_realsense_link_camera_0")}

    rows = []
    for phase, fs in FRAMES.items():
        for f in fs:
            st23 = extract_state_from_proprio(state61[f])
            gt = act[f:f + EXEC]
            stage = min(int(f / (n / NUM_STAGES)), NUM_STAGES - 1)
            obs = {
                "observation/egocentric_camera": resize_with_pad(imgs["zed_link_camera_0"][f], 224, 224),
                "observation/wrist_image_left": resize_with_pad(imgs["left_realsense_link_camera_0"][f], 224, 224),
                "observation/wrist_image_right": resize_with_pad(imgs["right_realsense_link_camera_0"][f], 224, 224),
                "observation/state": state61[f],
                "tokenized_prompt": np.array([TASK, stage], dtype=np.int32),
                "tokenized_prompt_mask": np.array([True, True]),
                "subtask_state": np.array(stage, dtype=np.int32),
            }
            for s in range(a.samples):
                pred = np.asarray(policy.infer(obs)["actions"])[:EXEC]
                r = {"label": a.label, "phase": phase, "frame": f, "stage": stage, "sample": s}
                for side, arm, g in (("L", L_ARM, L_GRIP), ("R", R_ARM, R_GRIP)):
                    dp, dg = pred[-1, arm] - st23[arm], gt[-1, arm] - st23[arm]
                    r[f"arm_err_{side}"] = float(np.abs(pred[:, arm] - gt[:, arm]).mean())
                    r[f"gt_motion_{side}"] = float(np.linalg.norm(dg))
                    r[f"cos_{side}"] = float(dp @ dg / (np.linalg.norm(dp) * np.linalg.norm(dg) + 1e-8))
                    r[f"mag_{side}"] = float(np.linalg.norm(dp) / (np.linalg.norm(dg) + 1e-8))
                    r[f"grip_{side}"] = float(pred[:, g].min())
                    r[f"gt_grip_{side}"] = float(gt[:, g].min())
                    r[f"gripmax_{side}"] = float(pred[:, g].max())
                    r[f"gt_gripmax_{side}"] = float(gt[:, g].max())
                    r[f"state_grip_{side}"] = float(st23[g])
                rows.append(r)
            last = rows[-a.samples:]
            med = lambda k: float(np.median([x[k] for x in last]))
            nopen = lambda side: sum(1 for x in last if x[f"gripmax_{side}"] > 0)
            print(f"{a.label:10s} {phase:22s} f{f:5d} st{stage} | state L {last[0]['state_grip_L']:+.2f} R {last[0]['state_grip_R']:+.2f} | "
                  f"PRED min/max L {med('grip_L'):+.2f}/{med('gripmax_L'):+.2f} (open {nopen('L')}/{a.samples}) "
                  f"R {med('grip_R'):+.2f}/{med('gripmax_R'):+.2f} (open {nopen('R')}/{a.samples}) | "
                  f"DEMO L {last[0]['gt_grip_L']:+.2f}/{last[0]['gt_gripmax_L']:+.2f} R {last[0]['gt_grip_R']:+.2f}/{last[0]['gt_gripmax_R']:+.2f}", flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"))
        print("saved", a.out)


if __name__ == "__main__":
    main()
