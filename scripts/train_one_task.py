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
CHECKPOINT_BASE = REPO_ROOT / "outputs/checkpoints/pi_behavior_b1k_fast_100task"


def main(task_index: int, ep_start: int, ep_end: int, weight_loader_params_path: str, exp_name: str, num_steps: int = NUM_STEPS, assets_base_dir: str | None = None, save_interval: int = 100000, keep_period: int | None = None):
    cfg = _config.get_config("pi_behavior_b1k_fast_100task")
    cfg = dataclasses.replace(
        cfg,
        exp_name=exp_name,
        num_train_steps=num_steps,
        **({"assets_base_dir": assets_base_dir} if assets_base_dir is not None else {}),
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
        model=dataclasses.replace(cfg.model, use_correlated_noise=False),
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
    params_path = f"{CHECKPOINT_BASE}/{exp_name}/{final_step}/params"
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
    p.add_argument("--assets-base-dir", type=str, default=None)
    p.add_argument("--save-interval", type=int, default=100000)
    p.add_argument("--keep-period", type=int, default=None)
    args = p.parse_args()
    main(args.task_index, args.ep_start, args.ep_end, args.weight_loader_params_path, args.exp_name, args.num_steps, args.assets_base_dir, args.save_interval, args.keep_period)
