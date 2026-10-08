"""Helper for watch_hf_and_eval.sh: python hf_changed_ckpts.py REPO BASELINE_REV
Prints "<head sha> <folders>" where <folders> lists every checkpoint folder (the dir holding params/manifest.ocdbt,
'.' for the repo root) whose manifest is new or changed vs BASELINE_REV and that also has assets/.../norm_stats.json;
'-' if none. With an empty BASELINE_REV every checkpoint folder in the head counts."""
import sys

from huggingface_hub import HfApi

a = HfApi()
repo, base = sys.argv[1], sys.argv[2]
sha = a.list_repo_commits(repo)[0].commit_id


def manifests(rev):
    out = {}
    for f in a.list_repo_tree(repo, revision=rev, recursive=True):
        if f.path.endswith("params/manifest.ocdbt"):
            out[f.path[: -len("params/manifest.ocdbt")].rstrip("/") or "."] = getattr(f, "blob_id", None)
    return out


files = set(a.list_repo_files(repo, revision=sha))
old = manifests(base) if base else {}
changed = []
for d, blob in manifests(sha).items():
    prefix = "" if d == "." else d + "/"
    has_stats = any(f.startswith(prefix + "assets/") and f.endswith("norm_stats.json") for f in files)
    if old.get(d) != blob and has_stats:
        changed.append(d)
print(sha, ",".join(sorted(changed)) or "-")
