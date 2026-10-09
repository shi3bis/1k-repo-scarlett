"""Offline metrics for one pushed checkpoint of the 76 -> 80 -> 87 chain,
appended to logs/push_evals_768087.{jsonl,md} and uploaded next to the
checkpoint on HF as eval_metrics.json.

Per evaluated task (own task, earlier chain tasks via --extra-tasks, and
--known-tasks the base was trained on, to catch forgetting):
  - teacher-forced training loss on a fixed sample of the task's episodes
    (eval_loss_one.py): loss, action_loss, per-joint action losses incl.
    both grippers, subtask_accuracy, fast_loss;
  - teacher-forced action error and exact / within-one stage accuracy from
    the full policy on one episode (eval_tf_one.py).
Each task runs in its own subprocess; a failure is recorded, not raised.

Only the episodes' own parquet/video files are fetched (LeRobot v3 packs
many episodes per file), and files this script downloaded are deleted
again at the end; files already on disk are left alone.
"""
import argparse
import datetime
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_ROOT = Path("/home/kasm-user/data/2026-challenge-demos")
DATASET_REPO = "behavior-1k/2026-challenge-demos"
LOG_DIR = Path("/workspace/persistent/behavior/logs")
# The dataset loader fetches the depth streams too if they're missing, so
# fetch (and later delete) them explicitly.
VIDEO_KEYS = [f"observation.{m}.{c}" for m in ("rgb", "depth_linear")
              for c in ("left_realsense_link_camera_0", "right_realsense_link_camera_0", "zed_link_camera_0")]
KNOWN_TASKS = [1, 35]  # in the base's training set (step-0 action loss 0.16 / subtask acc 0.95)


def episode_files(task_index: int, n: int) -> tuple[list[int], list[str]]:
    """First n episodes of a task and the data/video files that hold them."""
    if not (DATA_ROOT / "meta/episodes").exists():
        snapshot_download(repo_id=DATASET_REPO, repo_type="dataset", local_dir=str(DATA_ROOT), allow_patterns=["meta/**"])
    rows = []
    cols = ["episode_index", "task_index", "data/chunk_index", "data/file_index"]
    cols += [f"videos/{k}/{c}" for k in VIDEO_KEYS for c in ("chunk_index", "file_index")]
    for f in sorted(glob.glob(str(DATA_ROOT / "meta/episodes/*/*.parquet"))):
        t = pq.read_table(f, columns=cols).to_pylist()
        rows += [r for r in t if r["task_index"] == task_index]
    rows = sorted(rows, key=lambda r: r["episode_index"])[:n]
    files = set()
    for r in rows:
        files.add(f"data/chunk-{r['data/chunk_index']:03d}/file-{r['data/file_index']:03d}.parquet")
        for k in VIDEO_KEYS:
            files.add(f"videos/{k}/chunk-{r[f'videos/{k}/chunk_index']:03d}/file-{r[f'videos/{k}/file_index']:03d}.mp4")
    return [r["episode_index"] for r in rows], sorted(files)


def run_json(cmd: list[str], marker: str, log_path: Path) -> dict:
    with open(log_path, "a") as lf:
        lf.write(f"\n$ {' '.join(cmd)}\n")
        lf.flush()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              env=dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false"))
        lf.write(proc.stdout)
    for line in proc.stdout.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    return {"error": f"exit {proc.returncode}: " + " | ".join(proc.stdout.strip().splitlines()[-3:])}


def main(a):
    ckpt = Path(a.checkpoint_dir)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    sub_log = LOG_DIR / f"push_evals_{a.log_tag}_{a.label}.log"
    tasks = [("own", a.own_task)] if a.own_task is not None else []
    tasks += [("chain", t) for t in a.extra_tasks] + [("known", t) for t in a.known_tasks]

    downloaded = []
    plan = {}
    for _, t in tasks:
        eps, files = episode_files(t, a.loss_episodes)
        new = [f for f in files if not (DATA_ROOT / f).exists()]
        if new:
            snapshot_download(repo_id=DATASET_REPO, repo_type="dataset", local_dir=str(DATA_ROOT), allow_patterns=new)
            downloaded += new
        plan[t] = eps

    record = dict(label=a.label, checkpoint_dir=str(ckpt), hf_path=a.hf_path, time=stamp, tasks={})
    for role, t in tasks:
        eps = plan[t]
        loss = run_json([sys.executable, "-u", str(SCRIPT_DIR / "eval_loss_one.py"), "--params", str(ckpt / "params"),
                         "--task-index", str(t), "--episodes", *map(str, eps), "--num-batches", str(a.num_batches),
                         "--batch-size", str(a.batch_size)], "EVAL_LOSS_JSON:", sub_log)
        tf = run_json([sys.executable, "-u", str(SCRIPT_DIR / "eval_tf_one.py"), "--checkpoint-dir", str(ckpt),
                       "--task-index", str(t), "--episode-idx", str(eps[0]), "--stride", str(a.stride)], "EVAL_TF_JSON:", sub_log)
        record["tasks"][str(t)] = dict(role=role, loss=loss, tf=tf)
        print(f"[{a.label}] task {t} ({role}): loss={loss.get('loss', loss.get('error'))} tf_mae={tf.get('action_mae_chunk', tf.get('error'))}", flush=True)

    for f in downloaded:
        (DATA_ROOT / f).unlink(missing_ok=True)

    with open(LOG_DIR / f"push_evals_{a.log_tag}.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    md = LOG_DIR / f"push_evals_{a.log_tag}.md"
    if not md.exists():
        md.write_text("# Per-push offline evals, 76 -> 80 -> 87 chain\n\n"
                      "loss/action/grippers/subacc: teacher-forced training loss on a fixed sample (eval_loss_one.py). "
                      "TF MAE (raw units, full chunk) and stage accuracy: full policy on one episode, stride 50 (eval_tf_one.py).\n\n"
                      "| time | checkpoint | task (role) | loss | action | L grip | R grip | subacc | fast | TF MAE chunk | TF MAE first | TF grip L/R | stage exact | within-1 |\n"
                      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")

    def g(d, k, fmt="{:.4f}"):
        return fmt.format(d[k]) if k in d else "err"

    with open(md, "a") as f:
        for t, r in record["tasks"].items():
            lo, tf = r["loss"], r["tf"]
            grip = f"{g(tf, 'mae_chunk_left_gripper')}/{g(tf, 'mae_chunk_right_gripper')}"
            f.write(f"| {stamp} | {a.label} | {t} ({r['role']}) | {g(lo, 'loss')} | {g(lo, 'action_loss')} | "
                    f"{g(lo, 'action_loss_left_gripper')} | {g(lo, 'action_loss_right_gripper')} | {g(lo, 'subtask_accuracy', '{:.3f}')} | "
                    f"{g(lo, 'fast_loss')} | {g(tf, 'action_mae_chunk', '{:.5f}')} | {g(tf, 'action_mae_first', '{:.5f}')} | {grip} | "
                    f"{g(tf, 'exact_stage_accuracy', '{:.3f}')} | {g(tf, 'within_one_stage_accuracy', '{:.3f}')} |\n")

    if a.hf_path:
        out = ckpt / "eval_metrics.json"
        out.write_text(json.dumps(record, indent=1))
        try:
            HfApi().upload_file(path_or_fileobj=str(out), path_in_repo=f"{a.hf_path}/eval_metrics.json",
                                repo_id=a.hf_repo, commit_message=f"offline eval metrics for {a.label}")
            print(f"[{a.label}] uploaded eval_metrics.json to {a.hf_repo}/{a.hf_path}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{a.label}] eval_metrics.json upload FAILED: {e!r}", flush=True)
    print(f"EVAL_PUSH_DONE {a.label}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True, help="step dir with params/ (and assets/)")
    p.add_argument("--label", required=True)
    p.add_argument("--own-task", type=int, default=None)
    p.add_argument("--extra-tasks", type=int, nargs="*", default=[])
    p.add_argument("--known-tasks", type=int, nargs="*", default=KNOWN_TASKS)
    p.add_argument("--hf-repo", default="shi2bis/tasks_76_80_87")
    p.add_argument("--log-tag", default="768087", help="logs/push_evals_<tag>.{jsonl,md}")
    p.add_argument("--hf-path", default=None, help="checkpoint's folder in the HF repo; eval_metrics.json goes there")
    p.add_argument("--loss-episodes", type=int, default=6)
    p.add_argument("--num-batches", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--stride", type=int, default=50)
    main(p.parse_args())
