"""Push one finished step dir (params/ + assets/) to a HF folder, verify it, then write READY.

  python scripts/push_ready.py --checkpoint-dir <step dir> --repo shi2bis/x --folder y \
      --tasks 35 63 78 --step 4500 --recipe "..." --serve-with "..."
READY is written only after the params, norm_stats and FAST tokenizer are seen in the repo.
"""
import argparse
import datetime
import json

from huggingface_hub import HfApi

p = argparse.ArgumentParser()
p.add_argument("--checkpoint-dir", required=True)
p.add_argument("--repo", required=True)
p.add_argument("--folder", required=True)
p.add_argument("--tasks", type=int, nargs="+", required=True)
p.add_argument("--step", type=int, required=True)
p.add_argument("--recipe", required=True)
p.add_argument("--serve-with", required=True)
p.add_argument("--private", action="store_true")
a = p.parse_args()

api = HfApi()
api.create_repo(a.repo, private=a.private, exist_ok=True)
api.upload_folder(repo_id=a.repo, folder_path=a.checkpoint_dir, path_in_repo=a.folder,
                  allow_patterns=["params/**", "assets/**"], commit_message=f"{a.folder} (step {a.step}): {a.recipe}")
files = api.list_repo_files(a.repo)
n = sum(f.startswith(f"{a.folder}/params/") for f in files)
stats = any(f.startswith(f"{a.folder}/assets/") and f.endswith("norm_stats.json") for f in files)
tok = any(f.startswith(f"{a.folder}/assets/") and "fast_tokenizer" in f for f in files)
assert n and stats and tok, f"push verification FAILED: {n} params files, norm_stats={stats}, fast_tokenizer={tok}"
ready = dict(tasks=a.tasks, folder=a.folder, step=a.step, recipe=a.recipe, params_files=n,
             pushed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
             norm_stats="bundled under assets/ (Ilia's) + FAST tokenizer; decode with these",
             serve_with=a.serve_with)
api.upload_file(path_or_fileobj=json.dumps(ready, indent=1).encode(), path_in_repo=f"{a.folder}/READY",
                repo_id=a.repo, commit_message=f"READY: {a.folder}")
print(f"PUSHED + READY: {a.repo}/{a.folder} ({n} params files)", flush=True)
