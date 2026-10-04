"""Standalone task 0 (turning_on_radio) fine-tune: 2000 steps from Ilia's ext100
base, pushed to its own HF repo. Completely separate from the 76 -> 80 -> 87
chain: own branch (task0-radio-2000, off task35-3000-lowlr-decay, so the
ext100 stage table with 627 rows), own git worktree, own HF repo, own exp name.

Recipe: batch 32, LR warmup 200 -> 2e-5, cosine to 2e-6 by step 2000 (the
short-run schedule that beat the 20k-step default on task 35), Ilia's
norm_stats + FAST tokenizer, correlated noise on, 15 flow samples.

HF layout (same convention as the chain's repo, for the eval pod):
  task0_turning_on_radio/          final params/ + assets/, READY written last
  partial/task0_turning_on_radio/  latest intermediate save, never READY

Run from the worktree with PYTHONPATH=<worktree>/src so b1k resolves to this
branch's code, not the editable install in the main checkout.
"""
import datetime
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [T0] %(message)s")
log = logging.getLogger("task0")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_ROOT = Path("/home/kasm-user/data/2026-challenge-demos")
DATASET_REPO = "behavior-1k/2026-challenge-demos"
HF_REPO = "shi2bis/task0_turning_on_radio"
HF_FOLDER = "task0_turning_on_radio"
INIT_CKPT = "/workspace/persistent/behavior/checkpoints/behavior_50t_checkpoint_ext100/params"
EXP_NAME = "task0_radio_2000"
TASK_INDEX, CHUNK = 0, "000"
EP_START, EP_END = 0, 200
NUM_STEPS, WARMUP, PEAK_LR, END_LR = 2000, 200, 2e-5, 2e-6
BATCH_SIZE, NUM_WORKERS, SAVE_INTERVAL = 32, 64, 500
RECIPE = (f"{NUM_STEPS} steps from ext100 base, bs {BATCH_SIZE}, corr noise, Ilia stats, "
          f"LR warmup {WARMUP} -> {PEAK_LR:g} cosine -> {END_LR:g}")
RGB_KEYS = [
    "observation.rgb.left_realsense_link_camera_0",
    "observation.rgb.right_realsense_link_camera_0",
    "observation.rgb.zed_link_camera_0",
]


def download() -> None:
    log.info(f"downloading chunk-{CHUNK} ...")
    t0 = time.time()
    snapshot_download(
        repo_id=DATASET_REPO, repo_type="dataset", local_dir=str(DATA_ROOT),
        allow_patterns=["meta/**", f"data/chunk-{CHUNK}/**"] + [f"videos/{k}/chunk-{CHUNK}/**" for k in RGB_KEYS],
    )
    log.info(f"download done in {time.time()-t0:.0f}s")


def delete_data() -> None:
    removed = []
    for d in [DATA_ROOT / "data" / f"chunk-{CHUNK}", *(p / f"chunk-{CHUNK}" for p in (DATA_ROOT / "videos").glob("*"))]:
        if d.exists():
            shutil.rmtree(d)
            removed.append(str(d))
    log.info(f"deleted local dataset: {removed}")


def train() -> str:
    log.info(f"training: {RECIPE}; exp {EXP_NAME}")
    t0 = time.time()
    proc = subprocess.Popen(
        [
            sys.executable, "-u", str(SCRIPT_DIR / "train_one_task.py"),
            "--task-index", str(TASK_INDEX), "--ep-start", str(EP_START), "--ep-end", str(EP_END),
            "--weight-loader-params-path", INIT_CKPT, "--exp-name", EXP_NAME,
            "--num-steps", str(NUM_STEPS), "--corr-noise",
            "--batch-size", str(BATCH_SIZE), "--num-workers", str(NUM_WORKERS),
            "--warmup-steps", str(WARMUP), "--peak-lr", str(PEAK_LR),
            "--decay-steps", str(NUM_STEPS), "--decay-lr", str(END_LR),
            "--save-interval", str(SAVE_INTERVAL), "--keep-period", "100000",
            "--partial-hf-repo", HF_REPO, "--partial-hf-prefix", f"partial/{HF_FOLDER}",
        ],
        cwd=str(REPO_ROOT),
        env=dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false"),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    params_path = None
    for line in proc.stdout:
        sys.stdout.write(line)
        if line.startswith("FINAL_CHECKPOINT_PATH:"):
            params_path = line.split("FINAL_CHECKPOINT_PATH:", 1)[1].strip()
    if proc.wait() != 0:
        raise RuntimeError(f"train_one_task.py failed with exit code {proc.returncode}")
    if not params_path or not Path(params_path).exists():
        raise RuntimeError(f"expected checkpoint not found: {params_path}")
    log.info(f"training done in {time.time()-t0:.0f}s")
    return params_path


def push(params_path: str) -> None:
    api = HfApi()
    step_dir = Path(params_path).parent
    log.info(f"pushing {step_dir} -> {HF_REPO}/{HF_FOLDER}")
    for attempt in range(1, 4):
        try:
            api.upload_folder(repo_id=HF_REPO, folder_path=str(step_dir), path_in_repo=HF_FOLDER,
                              allow_patterns=["params/**", "assets/**"], commit_message=f"task 0 final: {RECIPE}")
            break
        except Exception as e:  # noqa: BLE001
            log.warning(f"push attempt {attempt}/3 failed: {e!r}")
            if attempt == 3:
                raise
            time.sleep(30)
    files = api.list_repo_files(HF_REPO)
    pushed_params = [f for f in files if f.startswith(f"{HF_FOLDER}/params/")]
    pushed_stats = [f for f in files if f.startswith(f"{HF_FOLDER}/assets/") and f.endswith("norm_stats.json")]
    if not pushed_params or not pushed_stats:
        raise RuntimeError(f"push verification FAILED: {len(pushed_params)} params files, {len(pushed_stats)} norm_stats")
    log.info(f"push verified: {len(pushed_params)} params files + norm_stats under {HF_FOLDER}")
    ready = dict(task_index=TASK_INDEX, task_name="turning_on_radio", folder=HF_FOLDER, num_steps=NUM_STEPS, recipe=RECIPE,
                 params_files=len(pushed_params), pushed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                 norm_stats="bundled under assets/ (Ilia's); decode with these",
                 serve_with="ext100 TASK_NUM_STAGES (627 rows; branch task0-radio-2000)")
    api.upload_file(path_or_fileobj=json.dumps(ready, indent=1).encode(), path_in_repo=f"{HF_FOLDER}/READY",
                    repo_id=HF_REPO, commit_message=f"READY: {HF_FOLDER}")
    log.info(f"READY marker written: {HF_FOLDER}/READY")
    if any(f.startswith(f"partial/{HF_FOLDER}/") for f in files):
        api.delete_folder(repo_id=HF_REPO, path_in_repo=f"partial/{HF_FOLDER}", commit_message="drop partials (final pushed)")
        log.info(f"removed partial/{HF_FOLDER}")


def main():
    log.info(f"=== task 0 (turning_on_radio) START: {RECIPE} ===")
    HfApi().create_repo(HF_REPO, private=True, exist_ok=True)
    download()
    final = train()
    push(final)
    delete_data()
    log.info(f"=== task 0 DONE, local checkpoint: {Path(final).parent} ===")
    print(f"TASK0_FINAL_STEP_DIR:{Path(final).parent}", flush=True)


if __name__ == "__main__":
    main()
