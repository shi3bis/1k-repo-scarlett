"""Sequential 76 -> 80 -> 87 chain (dispose_of_glass, clean_up_broken_glass,
collecting_aluminum_cans), continuing ONE set of weights task to task, same
download-train-push-delete loop as joint_loop.py but standalone: own HF
repo, own exp names, nothing imported from the place_inside_4_tasks pipeline.

Starts from behavior_50t_checkpoint_ext100_t80t87 (ext100 + stage-embedding
rows for 80/87, built by extend_stage_embeddings.py). Checkpoints from this
chain need THIS branch's TASK_NUM_STAGES to train or serve (82's stage
offset moved), not the place_inside_4_tasks branches'.

Ilia's training recipe throughout: his norm_stats + FAST tokenizer
(train_one_task.py defaults), correlated flow-matching noise ON (first time
in production -- previous chains ran with it off), 15 flow samples, FAST
aux loss 0.05, subtask loss 0.1 (config defaults).

LR: cosine with warmup = 10% of each task's steps, peak 2e-5 -> 2e-6 by the
task's last step (the schedule that beat the 20k-step default on task 35).
Each task restarts the schedule, since each task is a fresh train_one_task
process (params only; optimizer state isn't carried over).

Usage: python scripts/chain_76_80_87.py [--num-steps N] [--tasks 76 80 87] [--resume-from PARAMS]
(e.g. --tasks 76 to stop after the first task for eval; then
--tasks 80 87 --resume-from /home/kasm-user/chain768087_ckpts/task76/params)
"""
import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [CHAIN] %(message)s")
log = logging.getLogger("chain_76_80_87")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_ROOT = Path("/home/kasm-user/data/2026-challenge-demos")
DATASET_REPO = "behavior-1k/2026-challenge-demos"
HF_REPO = "shi2bis/tasks_76_80_87"
INIT_CKPT = "/home/kasm-user/checkpoints/behavior_50t_checkpoint_ext100_t80t87/params"
# Finished (pushed) checkpoints are moved here, on the overlay disk: /workspace
# (~28GB free) can't hold the previous task's checkpoint plus the one being
# trained (12.6GB each).
PARKED_DIR = Path("/home/kasm-user/chain768087_ckpts")
BATCH_SIZE = 32
NUM_WORKERS = 64
PEAK_LR, END_LR = 2e-5, 2e-6
WARMUP_FRAC = 0.10
RGB_KEYS = [
    "observation.rgb.left_realsense_link_camera_0",
    "observation.rgb.right_realsense_link_camera_0",
    "observation.rgb.zed_link_camera_0",
]

TASKS = [
    dict(task_index=76, name="dispose_of_glass", ep_start=15200, ep_end=15400),
    dict(task_index=80, name="clean_up_broken_glass", ep_start=16000, ep_end=16200),
    dict(task_index=87, name="collecting_aluminum_cans", ep_start=17400, ep_end=17600),
]


def chunk(task: dict) -> str:
    return f"{task['task_index']:03d}"


def download_task(task: dict) -> None:
    log.info(f"[task {task['task_index']}] downloading chunk-{chunk(task)} ...")
    t0 = time.time()
    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        local_dir=str(DATA_ROOT),
        allow_patterns=["meta/**", f"data/chunk-{chunk(task)}/**"] + [f"videos/{k}/chunk-{chunk(task)}/**" for k in RGB_KEYS],
    )
    log.info(f"[task {task['task_index']}] download done in {time.time()-t0:.0f}s")


def delete_task_data(task: dict) -> None:
    removed = []
    for d in [DATA_ROOT / "data" / f"chunk-{chunk(task)}", *(p / f"chunk-{chunk(task)}" for p in (DATA_ROOT / "videos").glob("*"))]:
        if d.exists():
            shutil.rmtree(d)
            removed.append(str(d))
    log.info(f"[task {task['task_index']}] deleted local dataset: {removed}")


def train_task(task: dict, init_params: str, num_steps: int, save_interval: int) -> str:
    exp_name = f"chain768087_task{task['task_index']}"
    warmup = max(1, round(WARMUP_FRAC * num_steps))
    log.info(f"[task {task['task_index']}] training {num_steps} steps (warmup {warmup}, bs {BATCH_SIZE}), weights from {init_params}")
    t0 = time.time()
    # Fresh subprocess per task (see train_one_task.py's docstring). Batch 32
    # needs ~85GB, above JAX's default 75% preallocation cap.
    proc = subprocess.Popen(
        [
            sys.executable, "-u",
            str(SCRIPT_DIR / "train_one_task.py"),
            "--task-index", str(task["task_index"]),
            "--ep-start", str(task["ep_start"]),
            "--ep-end", str(task["ep_end"]),
            "--weight-loader-params-path", init_params,
            "--exp-name", exp_name,
            "--num-steps", str(num_steps),
            "--corr-noise",
            "--batch-size", str(BATCH_SIZE),
            "--num-workers", str(NUM_WORKERS),
            "--warmup-steps", str(warmup),
            "--peak-lr", str(PEAK_LR),
            "--decay-steps", str(num_steps),
            "--decay-lr", str(END_LR),
            # Params-only crash checkpoints (max_to_keep=1). keep_period past
            # num_steps so the config's keep_period=2000 doesn't pin an extra
            # 12GB copy on /workspace.
            "--save-interval", str(save_interval),
            "--keep-period", "100000",
        ],
        cwd=str(REPO_ROOT),
        env=dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    params_path = None
    for line in proc.stdout:
        sys.stdout.write(line)
        if line.startswith("FINAL_CHECKPOINT_PATH:"):
            params_path = line.split("FINAL_CHECKPOINT_PATH:", 1)[1].strip()
    if proc.wait() != 0:
        raise RuntimeError(f"[task {task['task_index']}] train_one_task.py failed with exit code {proc.returncode}")
    if not params_path or not Path(params_path).exists():
        raise RuntimeError(f"[task {task['task_index']}] expected checkpoint not found: {params_path}")
    log.info(f"[task {task['task_index']}] training done in {time.time()-t0:.0f}s")
    return params_path


def push_checkpoint(task: dict, params_path: str, num_steps: int) -> None:
    api = HfApi()
    step_dir = Path(params_path).parent
    path_in_repo = f"task{task['task_index']}_{task['name']}"
    log.info(f"[task {task['task_index']}] pushing {step_dir} -> {HF_REPO}/{path_in_repo}")
    t0 = time.time()
    for attempt in range(1, 4):
        try:
            api.upload_folder(
                repo_id=HF_REPO,
                folder_path=str(step_dir),
                path_in_repo=path_in_repo,
                allow_patterns=["params/**", "assets/**"],
                commit_message=f"76->80->87 chain after task {task['task_index']} ({task['name']}), {num_steps} steps",
            )
            break
        except Exception as e:  # noqa: BLE001
            log.warning(f"[task {task['task_index']}] push attempt {attempt}/3 failed: {e!r}")
            if attempt == 3:
                raise
            time.sleep(30)
    log.info(f"[task {task['task_index']}] push done in {time.time()-t0:.0f}s")

    files = api.list_repo_files(HF_REPO)
    pushed_params = [f for f in files if f.startswith(path_in_repo + "/params/")]
    pushed_stats = [f for f in files if f.startswith(path_in_repo + "/assets/") and f.endswith("norm_stats.json")]
    if not pushed_params or not pushed_stats:
        raise RuntimeError(f"[task {task['task_index']}] push verification FAILED under {path_in_repo}: {len(pushed_params)} params files, {len(pushed_stats)} norm_stats")
    log.info(f"[task {task['task_index']}] push verified: {len(pushed_params)} params files + norm_stats under {path_in_repo}")


def main(num_steps: int, tasks: list[int], resume_from: str | None, save_interval: int):
    current = resume_from or INIT_CKPT
    prev_to_clean = None
    for task in TASKS:
        if task["task_index"] not in tasks:
            continue
        log.info(f"=== TASK {task['task_index']} ({task['name']}) START ===")
        download_task(task)
        final = train_task(task, current, num_steps, save_interval)
        push_checkpoint(task, final, num_steps)
        delete_task_data(task)
        # Park the pushed checkpoint on the overlay disk (next task's init).
        step_dir = Path(final).parent
        parked = PARKED_DIR / f"task{task['task_index']}"
        shutil.rmtree(parked, ignore_errors=True)
        parked.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(step_dir), str(parked))
        shutil.rmtree(step_dir.parent, ignore_errors=True)
        log.info(f"[task {task['task_index']}] parked checkpoint at {parked}")
        # The checkpoint we trained FROM is on HF and no longer needed
        # locally; never delete the base (prev_to_clean is None for it).
        if prev_to_clean is not None:
            shutil.rmtree(prev_to_clean, ignore_errors=True)
            log.info(f"[task {task['task_index']}] cleaned up superseded local checkpoint: {prev_to_clean}")
        prev_to_clean = parked
        current = str(parked / "params")
        log.info(f"=== TASK {task['task_index']} ({task['name']}) DONE, checkpoint: {current} ===")
    log.info(f"=== CHAIN COMPLETE. Final checkpoint: {current} ===")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num-steps", type=int, default=3000)
    p.add_argument("--tasks", type=int, nargs="+", default=[t["task_index"] for t in TASKS])
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--save-interval", type=int, default=500)
    a = p.parse_args()
    main(a.num_steps, a.tasks, a.resume_from, a.save_interval)
