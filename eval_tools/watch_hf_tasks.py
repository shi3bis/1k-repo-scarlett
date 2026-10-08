"""Notify-only HF watcher (2026-10-04) for the task 80 / 87 checkpoints of the 76->80->87 chain.

Push contract of the training pod (user, 2026-10-04): FIXED final folder names, a READY marker written only after a
verified push, and partial/intermediate saves under a separate partial/ path. So a checkpoint counts only when:
  - its folder is NOT under partial/ (any path component named 'partial' is ignored),
  - the folder name mentions task 80 or 87,
  - it has params/manifest.ocdbt + assets/.../norm_stats.json,
  - and a READY marker exists in that folder (<folder>/READY or <folder>/READY.<ext>).
Folders that were already READY when the watcher started are ignored. Also reports a new shi2bis model repo naming
80/87 that has a READY checkpoint. Exits (printing what is ready) as soon as anything qualifies.
    python watch_hf_tasks.py [poll_s=300] [max_h=16]
"""
import re
import sys
import time

from huggingface_hub import HfApi

a = HfApi()
REPO = "shi2bis/tasks_76_80_87"
POLL = int(sys.argv[1]) if len(sys.argv) > 1 else 300
MAXH = float(sys.argv[2]) if len(sys.argv) > 2 else 16
TASK = re.compile(r"(task_?|_)(80|87)(\D|$)")
READY = re.compile(r"(^|/)READY(\.[A-Za-z0-9]+)?$")


def ready_ckpts(repo):
    fs = [f for f in a.list_repo_files(repo) if "partial" not in f.split("/")[:-1]]
    out = set()
    for f in fs:
        if not f.endswith("params/manifest.ocdbt"):
            continue
        d = f[: -len("params/manifest.ocdbt")].rstrip("/")
        pre = d + "/" if d else ""
        has_stats = any(g.startswith(pre + "assets/") and g.endswith("norm_stats.json") for g in fs)
        has_ready = any(g.startswith(pre) and READY.search(g) and g.count("/") == pre.count("/") for g in fs)
        if has_stats and has_ready:
            out.add(d or ".")
    return out


def task_repos():
    return {m.id for m in a.list_models(author="shi2bis") if re.search(r"(80|87)", m.id) and m.id != REPO}


def stamp():
    return time.strftime("%H:%M:%S UTC", time.gmtime())


base = ready_ckpts(REPO)
base_repos = task_repos()
print(stamp(), "watching", REPO, "| READY folders now:", sorted(base), "| other 80/87 repos now:", sorted(base_repos), flush=True)
t0 = time.time()
while time.time() - t0 < MAXH * 3600:
    time.sleep(POLL)
    try:
        new = sorted(d for d in ready_ckpts(REPO) - base if TASK.search(d))
        newrepos = []
        for r in sorted(task_repos() - base_repos):
            rd = ready_ckpts(r)
            if rd:
                newrepos.append(f"{r}:{','.join(sorted(rd))}")
    except Exception as e:  # network hiccups: keep polling
        print(stamp(), "poll error", e, flush=True)
        continue
    if new or newrepos:
        head = a.list_repo_commits(REPO)[0].commit_id
        print(stamp(), "READY CHECKPOINT(S):", REPO, "@", head[:10], new, "| other repos:", newrepos, flush=True)
        sys.exit(0)
print(stamp(), "gave up after", MAXH, "h", flush=True)
