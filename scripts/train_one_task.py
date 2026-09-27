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


def main(task_index: int, ep_start: int, ep_end: int, weight_loader_params_path: str, exp_name: str, num_steps: int = NUM_STEPS, assets_base_dir: str = ILIA_ASSETS_BASE_DIR, corr_noise: bool = False, save_interval: int = 100000, keep_period: int | None = None):
    cfg = _config.get_config("pi_behavior_b1k_fast_100task")
    cfg = dataclasses.replace(
        cfg,
        exp_name=exp_name,
        num_train_steps=num_steps,
        assets_base_dir=assets_base_dir,
        **({"keep_period": keep_period} if keep_period is not None else {}),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
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
    _train.main(cfg)

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
    args = p.parse_args()
    main(args.task_index, args.ep_start, args.ep_end, args.weight_loader_params_path, args.exp_name, args.num_steps, args.assets_base_dir, args.corr_noise, args.save_interval, args.keep_period)
