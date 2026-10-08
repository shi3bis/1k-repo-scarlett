"""Notify-only HF watcher (2026-10-05): exits as soon as ANY new READY checkpoint folder appears in shi2bis/tasks_76_80_87
(user: a push adding another 2000 steps to the 76-80-87 chain is coming; folder name unknown), or any shi2bis repo
gains a READY folder it did not have when the watcher started (new repos included).
Same READY rules as watch_hf_tasks.py: not under partial/, params/manifest.ocdbt + assets/.../norm_stats.json + READY.
    python watch_hf_any_ready.py [poll_s=300] [max_h=24]
"""
import re
import sys
import time

from huggingface_hub import HfApi

a = HfApi()
POLL = int(sys.argv[1]) if len(sys.argv) > 1 else 300
MAXH = float(sys.argv[2]) if len(sys.argv) > 2 else 24
READY = re.compile(r"(^|/)READY(\.[A-Za-z0-9]+)?$")


def ready_ckpts(repo):
    fs = [f for f in a.list_repo_files(repo) if "partial" not in f.split("/")[:-1]]
    out = set()
    for f in fs:
        if not f.endswith("params/manifest.ocdbt"):
            continue
        d = f[: -len("params/manifest.ocdbt")].rstrip("/")
        pre = d + "/" if d else ""
        if any(g.startswith(pre + "assets/") and g.endswith("norm_stats.json") for g in fs) and any(
            g.startswith(pre) and READY.search(g) and g.count("/") == pre.count("/") for g in fs
        ):
            out.add(d or ".")
    return out


def snapshot():
    snap = {}
    for m in a.list_models(author="shi2bis"):
        try:
            snap[m.id] = ready_ckpts(m.id)
        except Exception:
            snap[m.id] = set()
    return snap


def stamp():
    return time.strftime("%H:%M:%S UTC", time.gmtime())


base = snapshot()
print(stamp(), "watching all shi2bis repos; READY now:", {k: sorted(v) for k, v in base.items() if v}, flush=True)
t0 = time.time()
while time.time() - t0 < MAXH * 3600:
    time.sleep(POLL)
    try:
        cur = snapshot()
    except Exception as e:
        print(stamp(), "poll error", e, flush=True)
        continue
    new = {r: sorted(v - base.get(r, set())) for r, v in cur.items() if v - base.get(r, set())}
    if new:
        print(stamp(), "NEW READY CHECKPOINT(S):", new, flush=True)
        sys.exit(0)
print(stamp(), "gave up after", MAXH, "h", flush=True)
