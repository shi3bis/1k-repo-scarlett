"""Overlay eval base paths on the demo base path for the same scene + task instance (2026-10-05).

    python path_overlay.py --task collecting_aluminum_cans --scene hotel_suite_large --inst 2 --demo-ep 17400 \
        --chunk 087 [--eval label=poses.jsonl ...] --out fig.png

Demo base path: the 2026 demos store no world pose, only base_qvel (robot frame, 61-d state[0:3]). It is integrated at the
recorded timestamps from the task's robot_start pose (2026 available_tasks.yaml). Eval paths come from the harness pose log
(<out>/poses/<task>_<inst>.jsonl). Object positions: the instance's tro_state file (demo) / the pose log at t=0 (eval).
Background: the scene's floor_trav_0.png (0.01 m/px, centered on the origin; row index grows with y).
Prints, per path: length, and the first step at which the base is within 1.0 m (xy) of each task object.
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import pyarrow.parquet as pq
import yaml

DS = "/workspace/persistent/BEHAVIOR-1K-v3.7/datasets"
DEMO = "/home/kasm-user/demos2026"
RES = 0.01
SKIP = ("floor.", "agent.", "bed.", "table.", "countertop.", "bookcase.", "cabinet.", "coffee_table.")


def quat_yaw(q):  # xyzw
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def robot_start(task):
    d = yaml.safe_load(open(f"{DS}/2026-challenge-task-instances/metadata/available_tasks.yaml"))[task][0]
    p, q = d["robot_start_position"], d["robot_start_orientation"]
    return p[0], p[1], quat_yaw(q)


def tro_objects(scene, task, inst):
    f = f"{DS}/2026-challenge-task-instances/scenes/{scene}/json/{scene}_task_{task}_instances/{scene}_task_{task}_0_{inst}_template-tro_state.json"
    out = {}
    for k, v in json.load(open(f)).items():
        p = v.get("root_link", {}).get("pos") if isinstance(v, dict) else None
        if p is not None and not k.startswith(SKIP):
            out[k] = p[:2]
    return out


ANNOT_TO_BDDL = {"trash_can": "ashcan", "broken_glass": "broken__glass", "can_of_soda": "can__of__soda",
                 "ice_bucket": "bucket", "radio": "radio_receiver"}


def annot_events(annot, objs):
    """Anchor events for fitting the demo start: (frame, candidate object xy list, label).
    Only objects still at their initial (tro_state) position count: "pick up from X" anchors on X, "place in/on X, C"
    and "press X" anchor on the container C / X; once an object has been picked up it is 'moved' and no longer used."""
    import re as _re
    evs, moved = [], set()
    for s in json.load(open(annot))["skill_annotation"]:
        if s["skill_type"][0] == "navigation":
            continue
        desc, ids = s["skill_description"][0], s["object_id"][0]
        tgt = ids[1] if desc.startswith("place") and len(ids) > 1 else ids[0]
        cat = _re.sub(r"_\d+$", "", tgt)
        key = ANNOT_TO_BDDL.get(cat, cat)
        if key not in moved:
            c = [np.array(p) for k, p in objs.items() if k.startswith(key + ".")]
            if c:
                evs.append((s["frame_duration"][0], c, f"{desc} {tgt}"))
        if desc.startswith("pick up"):
            k0 = ANNOT_TO_BDDL.get(_re.sub(r"_\d+$", "", ids[0]), _re.sub(r"_\d+$", "", ids[0]))
            # several instances (cans, shards, glasses): each pick-up takes a different, still-unmoved one, so keep
            # the category; a single-instance object (bin, bucket, radio) is gone from its initial spot after pick-up
            if sum(1 for k in objs if k.startswith(k0 + ".")) == 1:
                moved.add(k0)
    return evs


def demo_path(chunk, ep, task, evs=None):
    """Integrate robot-frame base_qvel. The demos did NOT start at the eval's robot_start (checked 2026-10-05: fitted
    starts are ~2.5 m away), so with annotated events the start (x, y, yaw) is fitted so that the base is closest to the
    object of each grasp/place event (residual = typical reach, ~0.7-0.9 m). Returns xs, ys, fitted start, residual."""
    fs = sorted(glob.glob(f"{DEMO}/data/chunk-{chunk}/*.parquet"))
    for f in fs:
        t = pq.read_table(f, columns=["episode_index", "observation.state", "timestamp"]).to_pandas()
        e = t[t.episode_index == ep]
        if len(e):
            break
    st = np.stack(e["observation.state"].values)
    ts = e["timestamp"].values
    x = y = yaw = 0.0
    q = [(0.0, 0.0)]
    for i in range(1, len(st)):
        dt = ts[i] - ts[i - 1]
        vx, vy, wz = st[i - 1, 0:3]
        x += (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
        y += (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt
        yaw += wz * dt
        q.append((x, y))
    q = np.array(q)
    if not evs:
        sx, sy, syaw = robot_start(task)
        R = np.array([[math.cos(syaw), -math.sin(syaw)], [math.sin(syaw), math.cos(syaw)]])
        p = q @ R.T + np.array([sx, sy])
        return p[:, 0], p[:, 1], (sx, sy, syaw), None
    best = None
    for y0 in np.linspace(-math.pi, math.pi, 361):
        R = np.array([[math.cos(y0), -math.sin(y0)], [math.sin(y0), math.cos(y0)]])
        pts = np.array([R @ q[f] for f, _, _ in evs])
        s = np.zeros(2)
        for _ in range(10):
            tg = np.array([min(c, key=lambda p: np.linalg.norm(p - (s + pt))) for pt, (_, c, _) in zip(pts, evs)])
            s = (tg - pts).mean(0)
        err = np.mean([min(np.linalg.norm(p - (s + pt)) for p in c) for pt, (_, c, _) in zip(pts, evs)])
        if best is None or err < best[0]:
            best = (err, y0, s)
    err, y0, s = best
    R = np.array([[math.cos(y0), -math.sin(y0)], [math.sin(y0), math.cos(y0)]])
    p = q @ R.T + s
    return p[:, 0], p[:, 1], (float(s[0]), float(s[1]), float(y0)), float(err)


def eval_path(f):
    xs, ys, ts, objs = [], [], [], {}
    for line in open(f):
        r = json.loads(line)
        xs.append(r["base"][0]); ys.append(r["base"][1]); ts.append(r["t"])
        if "objs" in r and not objs:
            objs = {k: v[:2] for k, v in r["objs"].items() if not k.startswith(SKIP)}
    return np.array(xs), np.array(ys), np.array(ts), objs


def first_near(xs, ys, steps, objs, r=1.0):
    out = {}
    for k, p in objs.items():
        d = np.hypot(xs - p[0], ys - p[1])
        i = np.argmax(d < r) if (d < r).any() else None
        out[k] = (int(steps[i]) if i is not None else None, round(float(d.min()), 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True); ap.add_argument("--scene", required=True)
    ap.add_argument("--inst", type=int, required=True); ap.add_argument("--chunk", required=True)
    ap.add_argument("--demo-ep", type=int); ap.add_argument("--eval", nargs="*", default=[])
    ap.add_argument("--annot"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    img = plt.imread(f"{DS}/behavior-1k-assets/scenes/{a.scene}/layout/floor_trav_0.png")
    H, W = img.shape[:2]
    ext = [-W / 2 * RES, W / 2 * RES, -H / 2 * RES, H / 2 * RES]
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(img, cmap="gray", origin="lower", extent=ext, alpha=0.6)
    objs = tro_objects(a.scene, a.task, a.inst)
    sx, sy, syaw = robot_start(a.task)
    allx, ally = [sx], [sy]
    if a.demo_ep is not None:
        evs = annot_events(a.annot, objs) if a.annot else None
        dx, dy, dstart, derr = demo_path(a.chunk, a.demo_ep, a.task, evs)
        ax.plot(dx, dy, color="tab:green", lw=2, label=f"demo ep{a.demo_ep} ({len(dx)} steps)")
        ax.plot(dstart[0], dstart[1], "^", color="tab:green", ms=11, mec="black")
        allx += list(dx); ally += list(dy)
        print(f"demo ep{a.demo_ep}: {len(dx)} steps, length {np.hypot(np.diff(dx), np.diff(dy)).sum():.1f} m, "
              f"start ({dstart[0]:.2f},{dstart[1]:.2f}, yaw {dstart[2]:.2f})"
              + (f" [fitted to {len(evs)} annotated events, mean residual {derr:.2f} m]" if derr is not None else " [yaml start]"))
        print("  first within 1 m (step, min dist):", first_near(dx, dy, np.arange(len(dx)), objs))
        for f0, _, lab in evs or []:
            ax.plot(dx[f0], dy[f0], "o", color="tab:green", ms=6)
            ax.annotate(f"{lab} @{f0}", (dx[f0], dy[f0]), fontsize=6, color="darkgreen")
    colors = ["tab:red", "tab:blue", "tab:orange", "tab:purple"]
    for i, spec in enumerate(a.eval):
        label, f = spec.split("=", 1)
        ex, ey, et, eobjs = eval_path(f)
        ax.plot(ex, ey, color=colors[i % 4], lw=1.2, alpha=0.85, label=f"{label} ({et[-1]} steps)")
        ax.plot(ex[0], ey[0], "^", color=colors[i % 4], ms=9)
        allx += list(ex); ally += list(ey)
        print(f"{label}: {et[-1]} steps, length {np.hypot(np.diff(ex), np.diff(ey)).sum():.1f} m, "
              f"start ({ex[0]:.2f},{ey[0]:.2f}) vs yaml start ({sx:.2f},{sy:.2f})")
        print("  first within 1 m (step, min dist):", first_near(ex, ey, et, eobjs or objs))
        if eobjs:
            dev = {k: round(math.dist(eobjs[k], objs[k]), 2) for k in eobjs if k in objs}
            print("  eval t=0 objects vs tro_state (m):", dev)
    for k, p in objs.items():
        ax.plot(p[0], p[1], "s", color="black", ms=6)
        ax.annotate(k.replace(".n.01", "").replace(".n.02", ""), p, fontsize=7)
    ax.plot(sx, sy, "*", color="gold", ms=16, mec="black", label="robot_start")
    ax.set_xlim(min(allx + [p[0] for p in objs.values()]) - 2, max(allx + [p[0] for p in objs.values()]) + 2)
    ax.set_ylim(min(ally + [p[1] for p in objs.values()]) - 2, max(ally + [p[1] for p in objs.values()]) + 2)
    ax.set_aspect("equal"); ax.legend(fontsize=8, loc="best"); ax.set_title(f"{a.task} inst {a.inst} ({a.scene})")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=110, bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()
