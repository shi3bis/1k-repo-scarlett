"""Standalone task 35 (attach_a_camera_to_a_tripod) fine-tune: 3000 steps
from Ilia's ext100 base, pushed to its own HF repo. Deliberately separate
from the place_inside_4_tasks pipeline (own repo, own exp name, no
joint_loop imports); it only reuses train_one_task.py for the training step.

Steps: download chunk-035 (data + rgb videos), train, push params/ +
assets/ to the repo root, verify, then delete the local checkpoint and
dataset.

Same normalization/settings as the production chain (Ilia's norm_stats +
FAST tokenizer via train_one_task.py's defaults; corr noise off).
"""
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [T35] %(message)s")
log = logging.getLogger("task35")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_ROOT = Path("/home/kasm-user/data/2026-challenge-demos")
DATASET_REPO = "behavior-1k/2026-challenge-demos"
HF_REPO = "shi2bis/task35_attach_a_camera_to_a_tripod"
INIT_CKPT = "/workspace/persistent/behavior/checkpoints/behavior_50t_checkpoint_ext100/params"
EXP_NAME = "task35_from_base_3000"
NUM_STEPS = 3000
TASK_INDEX = 35
CHUNK = "035"
EP_START, EP_END = 7000, 7200
RGB_KEYS = [
    "observation.rgb.left_realsense_link_camera_0",
    "observation.rgb.right_realsense_link_camera_0",
    "observation.rgb.zed_link_camera_0",
]


def download() -> None:
    log.info(f"downloading chunk-{CHUNK} ...")
    t0 = time.time()
    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        local_dir=str(DATA_ROOT),
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
    log.info(f"training {NUM_STEPS} steps on task {TASK_INDEX}, weights from {INIT_CKPT}, exp {EXP_NAME}")
    t0 = time.time()
    # Fresh subprocess (see train_one_task.py's docstring); stream its output
    # so progress is visible in the log, and keep the final path.
    proc = subprocess.Popen(
        [
            sys.executable, "-u",
            str(SCRIPT_DIR / "train_one_task.py"),
            "--task-index", str(TASK_INDEX),
            "--ep-start", str(EP_START),
            "--ep-end", str(EP_END),
            "--weight-loader-params-path", INIT_CKPT,
            "--exp-name", EXP_NAME,
            "--num-steps", str(NUM_STEPS),
        ],
        cwd=str(REPO_ROOT),
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
        raise RuntimeError(f"train_one_task.py failed with exit code {proc.returncode}")
    if not params_path or not Path(params_path).exists():
        raise RuntimeError(f"expected checkpoint not found: {params_path}")
    log.info(f"training done in {time.time()-t0:.0f}s")
    return params_path


def push(params_path: str) -> None:
    api = HfApi()
    step_dir = Path(params_path).parent
    log.info(f"pushing {step_dir} -> {HF_REPO}")
    t0 = time.time()
    for attempt in range(1, 4):
        try:
            api.upload_folder(
                repo_id=HF_REPO,
                folder_path=str(step_dir),
                allow_patterns=["params/**", "assets/**"],
                commit_message=f"task {TASK_INDEX} from ext100 base, {NUM_STEPS} steps",
            )
            break
        except Exception as e:  # noqa: BLE001
            log.warning(f"push attempt {attempt}/3 failed: {e!r}")
            if attempt == 3:
                raise
            time.sleep(30)
    log.info(f"push done in {time.time()-t0:.0f}s")

    files = api.list_repo_files(HF_REPO)
    pushed_params = [f for f in files if f.startswith("params/")]
    pushed_stats = [f for f in files if f.startswith("assets/") and f.endswith("norm_stats.json")]
    if not pushed_params or not pushed_stats:
        raise RuntimeError(f"push verification FAILED: {len(pushed_params)} params files, {len(pushed_stats)} norm_stats")
    log.info(f"push verified: {len(pushed_params)} params files + norm_stats")


def main():
    log.info(f"=== task {TASK_INDEX} {NUM_STEPS} steps from base START ===")
    download()
    params_path = train()
    push(params_path)
    # Only delete local copies once the push is verified.
    shutil.rmtree(Path(params_path).parent.parent, ignore_errors=True)
    log.info(f"deleted local output checkpoint {Path(params_path).parent.parent}")
    delete_data()
    log.info(f"=== task {TASK_INDEX} {NUM_STEPS} steps from base DONE ===")


if __name__ == "__main__":
    main()
