"""Train one task's worth of steps in a fresh interpreter (run via subprocess
from joint_loop.py). Isolating each task into its own process worked around a
"Instruction 'train' corresponds to no data!" ValueError from the `datasets`
library that reproduced deterministically for a *second* task's episode range
inside a long-lived process (twice, including one retry), but never
reproduced when the identical LeRobotDataset construction was run standalone
in a fresh process -- pointing at some in-process cache/state in `datasets`/
pyarrow that a fresh process sidesteps rather than a real data problem.

Prints the final checkpoint's params/ path as the LAST line of stdout.
"""
import dataclasses
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
import train as _train  # noqa: E402
from b1k.training import config as _config  # noqa: E402
from b1k.training import weight_loaders  # noqa: E402
from openpi.training import optimizer as _optimizer  # noqa: E402

NUM_STEPS = 1000
BATCH_SIZE = 8
NUM_WORKERS = 32
CONFIG_NAME = "pi_behavior_b1k_fast_100task"
CHECKPOINT_BASE = REPO_ROOT / "outputs/checkpoints" / CONFIG_NAME
# Ilia's shipped norm_stats.json (per-timestep action stats) + FAST tokenizer,
# laid out as assets_base_dir/config_name/asset_id (staged by
# ablation_forgetting.prepare_ilia_assets). Training in the same normalized
# action space the ext100 base was trained in is what the forgetting ablation
# showed matters (step-0 right-gripper loss 6.25 with our task-1 stats vs
# 0.0034 with Ilia's); it also re-enables the FAST aux loss, which silently
# never loaded from outputs/assets (no fast_tokenizer/ there).
ILIA_ASSETS_BASE_DIR = str(REPO_ROOT / "outputs/assets_ilia")


def main(task_index: int, ep_start: int, ep_end: int, weight_loader_params_path: str, exp_name: str, num_steps: int = NUM_STEPS, assets_base_dir: str = ILIA_ASSETS_BASE_DIR, corr_noise: bool = False, save_interval: int = 100000, keep_period: int | None = None, lr_schedule: dict | None = None, batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS, release_weight: float = 1.0, partial_hf_repo: str | None = None, partial_hf_prefix: str | None = None):
    cfg = _config.get_config("pi_behavior_b1k_fast_100task")
    cfg = dataclasses.replace(
        cfg,
        exp_name=exp_name,
        num_train_steps=num_steps,
        assets_base_dir=assets_base_dir,
        **({"keep_period": keep_period} if keep_period is not None else {}),
        # Overrides only the given CosineDecaySchedule fields; None keeps the
        # config's schedule (warmup 1000, peak 1e-4, decay over 20k steps).
        **({"lr_schedule": dataclasses.replace(cfg.lr_schedule, **lr_schedule)} if lr_schedule else {}),
        batch_size=batch_size,
        num_workers=num_workers,
        log_interval=10,
        save_interval=save_interval,
        wandb_enabled=False,
        overwrite=True,
        seed=0,
        save_train_state=False,
        weight_loader=weight_loaders.PiBehaviorWeightLoader(weight_loader_params_path),
        model=dataclasses.replace(cfg.model, use_correlated_noise=corr_noise),
        data=dataclasses.replace(
            cfg.data,
            base_config=dataclasses.replace(
                cfg.data.base_config,
                episodes_index=list(range(ep_start, ep_end)),
            ),
        ),
    )
    on_saved = None
    if partial_hf_repo and partial_hf_prefix:
        def on_saved(step, step_dir):
            # Intermediate (crash-recovery) saves go to a separate partial/ path on
            # HF with no READY marker, so the eval pod never mistakes them for a
            # finished checkpoint; only the latest one is kept there. The final
            # step is pushed by the caller into the task's fixed folder instead.
            if step == num_steps - 1:
                return
            from huggingface_hub import HfApi
            api = HfApi()
            dest = f"{partial_hf_prefix}/step{step}"
            try:
                api.upload_folder(repo_id=partial_hf_repo, folder_path=str(step_dir), path_in_repo=dest,
                                  allow_patterns=["params/**", "assets/**"], commit_message=f"partial save step {step} (not final)")
                files = api.list_repo_files(partial_hf_repo)
                old = sorted({f.split("/")[len(partial_hf_prefix.split("/"))] for f in files
                              if f.startswith(partial_hf_prefix + "/step")} - {f"step{step}"})
                for o in old:
                    api.delete_folder(repo_id=partial_hf_repo, path_in_repo=f"{partial_hf_prefix}/{o}", commit_message=f"drop older partial {o}")
                print(f"PARTIAL_PUSHED:{partial_hf_repo}/{dest} (removed older: {old})", flush=True)
            except Exception as e:  # noqa: BLE001 -- never kill training over a partial push
                print(f"PARTIAL_PUSH_FAILED step {step}: {e!r}", flush=True)
    _train.main(cfg, on_checkpoint_saved=on_saved, release_weight=release_weight)

    final_step = num_steps - 1
    step_dir = CHECKPOINT_BASE / exp_name / str(final_step)
    # save_state() only looks for fast_tokenizer/ under the default
    # outputs/assets path, so bundle the one we actually trained with; the
    # eval side needs this checkpoint's own assets (norm_stats + tokenizer).
    asset_id = cfg.data.assets.asset_id or cfg.data.repo_id
    src_tok = Path(assets_base_dir) / CONFIG_NAME / asset_id / "fast_tokenizer"
    dst_tok = step_dir / "assets" / asset_id / "fast_tokenizer"
    if src_tok.exists() and not dst_tok.exists():
        shutil.copytree(src_tok, dst_tok)
    params_path = step_dir / "params"
    print(f"FINAL_CHECKPOINT_PATH:{params_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--task-index", type=int, required=True)
    p.add_argument("--ep-start", type=int, required=True)
    p.add_argument("--ep-end", type=int, required=True)
    p.add_argument("--weight-loader-params-path", type=str, required=True)
    p.add_argument("--exp-name", type=str, required=True)
    p.add_argument("--num-steps", type=int, default=NUM_STEPS)
    p.add_argument("--assets-base-dir", type=str, default=ILIA_ASSETS_BASE_DIR)
    p.add_argument("--corr-noise", action="store_true")
    p.add_argument("--save-interval", type=int, default=100000)
    p.add_argument("--keep-period", type=int, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--peak-lr", type=float, default=None)
    p.add_argument("--decay-steps", type=int, default=None)
    p.add_argument("--decay-lr", type=float, default=None)
    # Batch 32 / 64 workers measured 2026-10-03: 3.72 s/step = 8.6 samples/s
    # (vs ~3-4 samples/s at 8/32, which was dataloader-starved), peak VRAM
    # ~85GB of 98GB -- 64 won't fit. Needs XLA_PYTHON_CLIENT_PREALLOCATE=false
    # (default preallocation caps JAX at 75% of VRAM).
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    # >1: weight samples whose action chunk contains a gripper release
    # (closed -> open) this many times in the batch-mean loss (train.release_mask).
    p.add_argument("--release-weight", type=float, default=1.0)
    p.add_argument("--partial-hf-repo", type=str, default=None)
    p.add_argument("--partial-hf-prefix", type=str, default=None)
    args = p.parse_args()
    lr_schedule = {k: v for k, v in dict(warmup_steps=args.warmup_steps, peak_lr=args.peak_lr, decay_steps=args.decay_steps, decay_lr=args.decay_lr).items() if v is not None}
    main(args.task_index, args.ep_start, args.ep_end, args.weight_loader_params_path, args.exp_name, args.num_steps, args.assets_base_dir, args.corr_noise, args.save_interval, args.keep_period, lr_schedule, args.batch_size, args.num_workers, args.release_weight, args.partial_hf_repo, args.partial_hf_prefix)
