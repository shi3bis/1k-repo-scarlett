"""2026-10-07: exit as soon as any shi2bis repo gets a NEW commit whose title or new files mention task0 / radio / 5000
(catches a 5000-step task-0 push even without a READY marker). Poll 5 min, max 30 h."""
import sys, time, re
from huggingface_hub import HfApi
a = HfApi(); pat = re.compile(r"task ?0\b|task0|radio|5000", re.I)
def heads():
    return {m.id: a.list_repo_commits(m.id)[0].commit_id for m in a.list_models(author="shi2bis")}
base = heads(); print(time.strftime("%H:%M:%S UTC", time.gmtime()), "watching commits of", sorted(base), flush=True)
t0 = time.time()
while time.time() - t0 < 30 * 3600:
    time.sleep(300)
    try:
        cur = heads()
        for r, h in cur.items():
            if base.get(r) == h: continue
            new = []
            for c in a.list_repo_commits(r):
                if c.commit_id == base.get(r): break
                new.append(c)
            hits = [c for c in new if pat.search(c.title or "")]
            files = [f for f in a.list_repo_files(r) if pat.search(f)] if not hits else []
            if hits or (r.endswith("task0_turning_on_radio") and new) or files:
                print(time.strftime("%H:%M:%S UTC", time.gmtime()), "NEW TASK-0 COMMIT(S) in", r, [(c.commit_id[:10], c.title) for c in new[:8]], "files:", files[:10], flush=True)
                sys.exit(0)
            base[r] = h
    except Exception as e:
        print("poll error", e, flush=True)
print("gave up", flush=True)
