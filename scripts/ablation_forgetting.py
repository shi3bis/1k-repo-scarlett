"""Catastrophic-forgetting ablation: 4 arms, all trained on task 1
(picking_up_trash) starting from behavior_50t_checkpoint_ext100, each saving
checkpoints at steps 10/25/100/300 for closed-loop eval on a separate
OmniGibson pod.

Arms:
  baseline             our task-1 2026 norm_stats, no correlated noise, default LR
                       (= the recipe every production fine-tune so far used)
  iliastats            Ilia's own norm_stats (per-timestep action norm) + his
                       FAST tokenizer (re-enables the FAST aux loss, silently
                       off in all our runs), no correlated noise
  iliastats_corrnoise  as iliastats, plus use_correlated_noise=True with Ilia's
                       960x960 action Cholesky -- i.e. Ilia's full recipe
  lowlr_w1000          as baseline, but peak_lr=1e-5 / warmup 1000 (default:
                       1e-4 / warmup 1000) -- 10x lower LR at every step
  lowlr                (superseded) peak_lr=1e-5 / warmup 100. Its linear ramp
                       is lr = 1e-7*step, IDENTICAL to the default schedule's
                       ramp through step 100, so its step10/25/100 checkpoints
                       are effectively baseline duplicates; kept only so the
                       existing HF folder stays traceable.

Each checkpoint (params + assets, incl. the norm_stats that arm trained with --
the eval side MUST use these, not another arm's) is pushed to
shi2bis/place_inside_4_tasks under ablations_forgetting/<arm>/step<N>/ and
deleted locally before training continues (~12GB each; disk can't hold a
sweep).

Usage:
  python scripts/ablation_forgetting.py                 # all non-superseded arms, sequentially
  python scripts/ablation_forgetting.py baseline lowlr  # a subset
Each arm runs in a fresh subprocess (see train_one_task.py docstring for why).
"""
import dataclasses
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ABLATION] %(message)s")
log = logging.getLogger("ablation")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_ROOT = Path("/home/kasm-user/data/2026-challenge-demos")
DATA_REPO_ID = "behavior-1k/2026-challenge-demos"
HF_REPO = "shi2bis/place_inside_4_tasks"
HF_PREFIX = "ablations_forgetting"
BASE_CKPT = "/workspace/persistent/behavior/checkpoints/behavior_50t_checkpoint_ext100/params"
CONFIG_NAME = "pi_behavior_b1k_fast_100task"
ASSET_ID = DATA_REPO_ID

TASK = dict(task_index=1, chunk="001", ep_start=200, ep_end=400)
SAVE_STEPS = (10, 25, 100, 300)
NUM_STEPS = max(SAVE_STEPS) + 1  # loop step index 300 must exist
BATCH_SIZE = 8
NUM_WORKERS = 32

OUR_ASSETS = REPO_ROOT / "outputs/assets"
ILIA_ASSETS = REPO_ROOT / "outputs/assets_ilia"
ILIA_HF_REPO = "IliaLarchenko/behavior_50t_checkpoint"
ILIA_HF_ASSET_PREFIX = "assets/IliaLarchenko/behavior_224_rgb"

ARMS = {
    "baseline": dict(assets=OUR_ASSETS, corr_noise=False, lr=None),
    "iliastats": dict(assets=ILIA_ASSETS, corr_noise=False, lr=None),
    "iliastats_corrnoise": dict(assets=ILIA_ASSETS, corr_noise=True, lr=None),
    "lowlr_w1000": dict(assets=OUR_ASSETS, corr_noise=False, lr=dict(warmup_steps=1000, peak_lr=1e-5, decay_lr=1e-6)),
    "lowlr": dict(assets=OUR_ASSETS, corr_noise=False, lr=dict(warmup_steps=100, peak_lr=1e-5, decay_lr=1e-6)),
}


def ilia_asset_dir() -> Path:
    return ILIA_ASSETS / CONFIG_NAME / ASSET_ID


def prepare_ilia_assets() -> None:
    """Lay Ilia's shipped norm_stats.json + fast_tokenizer/ out under the
    assets_base_dir/config_name/asset_id layout DataConfigFactory reads."""
    from huggingface_hub import snapshot_download

    dest = ilia_asset_dir()
    if (dest / "norm_stats.json").exists() and (dest / "fast_tokenizer").exists():
        return
    staging = REPO_ROOT / "outputs/.ilia_assets_download"
    snapshot_download(ILIA_HF_REPO, allow_patterns=[f"{ILIA_HF_ASSET_PREFIX}/**"], local_dir=str(staging))
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(staging / ILIA_HF_ASSET_PREFIX / "norm_stats.json", dest / "norm_stats.json")
    shutil.copytree(staging / ILIA_HF_ASSET_PREFIX / "fast_tokenizer", dest / "fast_tokenizer", dirs_exist_ok=True)
    shutil.rmtree(staging)
    log.info(f"Ilia assets staged at {dest}")


def download_task1() -> None:
    from huggingface_hub import snapshot_download

    chunk = TASK["chunk"]
    log.info(f"downloading chunk-{chunk} ...")
    snapshot_download(
        repo_id=DATA_REPO_ID,
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


def push_and_delete(arm: str, step: int, step_dir: Path, assets_base: Path) -> None:
    from huggingface_hub import HfApi

    # save_state() only copies fast_tokenizer from the default outputs/assets
    # path, so for Ilia-assets arms bundle it here explicitly -- the eval side
    # needs the exact assets this arm trained with.
    src_tok = assets_base / CONFIG_NAME / ASSET_ID / "fast_tokenizer"
    dst_tok = step_dir / "assets" / ASSET_ID / "fast_tokenizer"
    if src_tok.exists() and not dst_tok.exists():
        shutil.copytree(src_tok, dst_tok)

    api = HfApi()
    path_in_repo = f"{HF_PREFIX}/{arm}/step{step}"
    t0 = time.time()
    for attempt in range(1, 4):
        try:
            api.upload_folder(
                repo_id=HF_REPO,
                folder_path=str(step_dir),
                path_in_repo=path_in_repo,
                allow_patterns=["params/**", "assets/**"],
                commit_message=f"forgetting ablation: {arm} step {step} (task 1)",
            )
            break
        except Exception as e:  # noqa: BLE001 -- transient xet timeouts seen before
            log.warning(f"[{arm}] push step {step} attempt {attempt}/3 failed: {e!r}")
            if attempt == 3:
                raise
            time.sleep(30)
    pushed = [f for f in api.list_repo_files(HF_REPO) if f.startswith(path_in_repo + "/params/")]
    if not pushed:
        raise RuntimeError(f"[{arm}] push verification FAILED for {path_in_repo}")
    log.info(f"[{arm}] step {step} pushed ({len(pushed)} params files, {time.time()-t0:.0f}s) -> {path_in_repo}")
    shutil.rmtree(step_dir)


def run_arm(arm: str) -> None:
    """Runs inside a fresh subprocess."""
    sys.path.insert(0, str(SCRIPT_DIR))
    import train as _train
    from b1k.training import config as _config
    from b1k.training import weight_loaders
    import openpi.training.optimizer as _optimizer

    spec = ARMS[arm]
    cfg = _config.get_config(CONFIG_NAME)
    lr_schedule = cfg.lr_schedule
    if spec["lr"] is not None:
        lr_schedule = dataclasses.replace(cfg.lr_schedule, **spec["lr"])
    cfg = dataclasses.replace(
        cfg,
        exp_name=f"ablation_{arm}",
        num_train_steps=NUM_STEPS,
        save_steps=SAVE_STEPS,
        save_interval=10**9,
        keep_period=1,  # never let orbax prune; push_and_delete owns deletion
        assets_base_dir=str(spec["assets"]),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        log_interval=5,
        wandb_enabled=False,
        overwrite=True,
        seed=0,
        save_train_state=False,
        lr_schedule=lr_schedule,
        weight_loader=weight_loaders.PiBehaviorWeightLoader(BASE_CKPT),
        model=dataclasses.replace(cfg.model, use_correlated_noise=spec["corr_noise"]),
        data=dataclasses.replace(
            cfg.data,
            base_config=dataclasses.replace(
                cfg.data.base_config,
                episodes_index=list(range(TASK["ep_start"], TASK["ep_end"])),
            ),
        ),
    )
    assert isinstance(cfg.lr_schedule, _optimizer.CosineDecaySchedule)
    log.info(f"[{arm}] lr_schedule={cfg.lr_schedule} corr_noise={spec['corr_noise']} assets={spec['assets']}")
    _train.main(cfg, on_checkpoint_saved=lambda step, d: push_and_delete(arm, step, Path(d), spec["assets"]))


def main(arms: list[str]) -> None:
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}; choose from {list(ARMS)}")
    prepare_ilia_assets()
    download_task1()
    for arm in arms:
        log.info(f"=== ARM {arm} START ===")
        t0 = time.time()
        rc = subprocess.run([sys.executable, "-u", __file__, "--run-arm", arm], cwd=str(REPO_ROOT)).returncode
        if rc != 0:
            raise RuntimeError(f"arm {arm} failed with exit code {rc}")
        log.info(f"=== ARM {arm} DONE in {time.time()-t0:.0f}s ===")
    log.info("=== ALL ARMS DONE ===")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--run-arm":
        run_arm(sys.argv[2])
    else:
        main(sys.argv[1:] or [a for a in ARMS if a != "lowlr"])
