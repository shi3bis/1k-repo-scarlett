"""Download-train-push-delete loop across the 4 selected BEHAVIOR Challenge 2026
tasks (1, 55, 76, 82), continuing ONE set of weights sequentially task to task.

Per task:
  1. Download that task's full 200-episode chunk (data + rgb videos; meta is
     already fully present locally).
  2. Train `NUM_STEPS` steps from the previous task's saved checkpoint
     (task 1 starts from checkpoints/behavior_50t_checkpoint_ext100).
  3. Push the resulting `params/` checkpoint (not the full train_state --
     smaller, and it's the same format PiBehaviorWeightLoader reads) to
     shi2bis/place_inside_4_tasks on Hugging Face.
  4. Verify the push landed, THEN delete that task's local dataset.
  5. Move to the next task, continuing from the checkpoint just pushed.

Known simplifications (flagged in the final report, not silently hidden):
  - use_correlated_noise stays disabled (same as the smoke test) --
    action_correlation_cholesky was never computed against the 2026 schema.
  - norm_stats reused from task 1 only, not recomputed per task or combined
    across all 4 -- state/action ranges are robot-level (same R1Pro
    embodiment), not task-specific, so this is an approximation, not
    normalizing against literal zeros.
  - This is sequential curriculum fine-tuning (one task's data at a time,
    per the disk-driven delete-after-use design), not batch-mixed joint
    training across all 4 tasks' data simultaneously.
"""
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [LOOP] %(message)s")
log = logging.getLogger("joint_loop")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_ROOT = Path("/home/kasm-user/data/2026-challenge-demos")
REPO_ID = "behavior-1k/2026-challenge-demos"
HF_PROGRESS_REPO = "shi2bis/place_inside_4_tasks"
NUM_STEPS = 1000

TASKS = [
    dict(task_index=1, name="picking_up_trash", chunk="001", ep_start=200, ep_end=400),
    dict(task_index=55, name="re_shelving_library_books", chunk="055", ep_start=11000, ep_end=11200),
    dict(task_index=76, name="dispose_of_glass", chunk="076", ep_start=15200, ep_end=15400),
    dict(task_index=82, name="store_batteries", chunk="082", ep_start=16400, ep_end=16600),
]


def download_task(task: dict) -> None:
    chunk = task["chunk"]
    log.info(f"[task {task['task_index']}] downloading chunk-{chunk} ...")
    t0 = time.time()
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(DATA_ROOT),
        allow_patterns=[
            "meta/**",
            f"data/chunk-{chunk}/**",
            f"videos/observation.rgb.left_realsense_link_camera_0/chunk-{chunk}/**",
            f"videos/observation.rgb.right_realsense_link_camera_0/chunk-{chunk}/**",
            f"videos/observation.rgb.zed_link_camera_0/chunk-{chunk}/**",
        ],
    )
    log.info(f"[task {task['task_index']}] download done in {time.time()-t0:.0f}s")


def delete_task_data(task: dict) -> None:
    chunk = task["chunk"]
    removed = []
    data_dir = DATA_ROOT / "data" / f"chunk-{chunk}"
    if data_dir.exists():
        shutil.rmtree(data_dir)
        removed.append(str(data_dir))
    for video_key_dir in (DATA_ROOT / "videos").glob("*"):
        chunk_dir = video_key_dir / f"chunk-{chunk}"
        if chunk_dir.exists():
            shutil.rmtree(chunk_dir)
            removed.append(str(chunk_dir))
    log.info(f"[task {task['task_index']}] deleted local dataset: {removed}")


def train_task(task: dict, weight_loader_params_path: str) -> str:
    # Run in a fresh subprocess, not in-process: a second task's training in
    # this same long-lived process deterministically hit "Instruction 'train'
    # corresponds to no data!" from the `datasets` library (twice, including
    # a same-process retry) while the identical LeRobotDataset construction
    # succeeded standalone in a fresh interpreter -- points at in-process
    # cache/state in `datasets`/pyarrow, not an actual data problem. A fresh
    # process per task sidesteps whatever that state is.
    exp_name = f"joint_task{task['task_index']}"
    log.info(f"[task {task['task_index']}] training {NUM_STEPS} steps, weights from {weight_loader_params_path}")
    t0 = time.time()
    result = subprocess.run(
        [
            sys.executable, "-u",
            str(SCRIPT_DIR / "train_one_task.py"),
            "--task-index", str(task["task_index"]),
            "--ep-start", str(task["ep_start"]),
            "--ep-end", str(task["ep_end"]),
            "--weight-loader-params-path", weight_loader_params_path,
            "--exp-name", exp_name,
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"[task {task['task_index']}] train_one_task.py failed with exit code {result.returncode}")
    log.info(f"[task {task['task_index']}] training done in {time.time()-t0:.0f}s")

    final_line = [l for l in result.stdout.splitlines() if l.startswith("FINAL_CHECKPOINT_PATH:")]
    if not final_line:
        raise RuntimeError(f"[task {task['task_index']}] train_one_task.py did not report a final checkpoint path")
    params_path = final_line[-1].split("FINAL_CHECKPOINT_PATH:", 1)[1].strip()
    if not Path(params_path).exists():
        raise RuntimeError(f"[task {task['task_index']}] expected checkpoint not found: {params_path}")
    return params_path


def push_checkpoint(task: dict, params_path: str) -> None:
    api = HfApi()
    path_in_repo = f"task{task['task_index']}_{task['name']}/params"
    log.info(f"[task {task['task_index']}] pushing {params_path} -> {HF_PROGRESS_REPO}/{path_in_repo}")
    t0 = time.time()
    api.create_repo(HF_PROGRESS_REPO, exist_ok=True)
    api.upload_folder(
        repo_id=HF_PROGRESS_REPO,
        folder_path=params_path,
        path_in_repo=path_in_repo,
        commit_message=f"joint checkpoint after task {task['task_index']} ({task['name']}), {NUM_STEPS} steps",
    )
    log.info(f"[task {task['task_index']}] push done in {time.time()-t0:.0f}s")

    # Verify: the pushed folder must actually contain files, not just an empty commit.
    files = api.list_repo_files(HF_PROGRESS_REPO)
    pushed_files = [f for f in files if f.startswith(path_in_repo + "/")]
    if not pushed_files:
        raise RuntimeError(f"[task {task['task_index']}] push verification FAILED: no files found under {path_in_repo}")
    log.info(f"[task {task['task_index']}] push verified: {len(pushed_files)} files under {path_in_repo}")


def train_task_with_retry(task: dict, weight_loader_params_path: str, attempts: int = 2) -> str:
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return train_task(task, weight_loader_params_path)
        except Exception as e:  # noqa: BLE001
            last_exc = e
            log.warning(
                f"[task {task['task_index']}] training attempt {attempt}/{attempts} failed: {e!r}"
                + (" -- retrying" if attempt < attempts else "")
            )
            if attempt < attempts:
                time.sleep(10)
    raise last_exc


def main(start_task_index: int | None = None, resume_from_ckpt: str | None = None):
    current_ckpt = resume_from_ckpt or "/workspace/persistent/behavior/checkpoints/behavior_50t_checkpoint_ext100/params"
    prev_ckpt_to_clean = None
    for task in TASKS:
        if start_task_index is not None and task["task_index"] < start_task_index:
            continue
        log.info(f"=== TASK {task['task_index']} ({task['name']}) START ===")
        download_task(task)
        final_ckpt = train_task_with_retry(task, current_ckpt)
        push_checkpoint(task, final_ckpt)
        delete_task_data(task)
        # The checkpoint we trained FROM is already safely on HF (pushed in
        # an earlier iteration, or -- for task 1 -- is the pre-existing base
        # checkpoint we never delete) and is no longer needed locally once
        # this task's own checkpoint exists; free its disk space.
        if prev_ckpt_to_clean is not None:
            shutil.rmtree(Path(prev_ckpt_to_clean).parent, ignore_errors=True)
            log.info(f"[task {task['task_index']}] cleaned up superseded local checkpoint: {prev_ckpt_to_clean}")
        prev_ckpt_to_clean = final_ckpt
        current_ckpt = final_ckpt
        log.info(f"=== TASK {task['task_index']} ({task['name']}) DONE, checkpoint: {current_ckpt} ===")
    log.info(f"=== LOOP COMPLETE. Final checkpoint: {current_ckpt} ===")


if __name__ == "__main__":
    import sys as _sys

    start_idx = int(_sys.argv[1]) if len(_sys.argv) > 1 else None
    resume_ckpt = _sys.argv[2] if len(_sys.argv) > 2 else None
    main(start_task_index=start_idx, resume_from_ckpt=resume_ckpt)
