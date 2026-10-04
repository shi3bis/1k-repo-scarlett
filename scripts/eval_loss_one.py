"""Teacher-forced training-loss metrics of one checkpoint on one task's data
(run as a subprocess from eval_push.py, one task per process -- a second
LeRobotDataset construction in the same process hits the `datasets`
"Instruction 'train' corresponds to no data!" bug, see train_one_task.py).

Runs the exact training loss (compute_detailed_loss, train=True so the
subtask head is scored, same num_flow_samples / corr noise / Ilia stats as
training) on a FIXED set of batches: seed 0 for the sampler and a fixed
flow-noise rng per batch, so numbers are comparable across checkpoints.
No gradient step is taken.

Prints one line `EVAL_LOSS_JSON:{...}` with the mean of every scalar metric
(loss, action_loss, action_loss_<joint> incl. grippers, subtask_accuracy,
fast_loss, ...).
"""
import dataclasses
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from flax import nnx  # noqa: E402

import train as _train  # noqa: E402
from b1k.training import config as _config  # noqa: E402
from b1k.training import data_loader as _data_loader  # noqa: E402
from b1k.training import weight_loaders  # noqa: E402
from openpi.training import sharding  # noqa: E402
from train_one_task import ILIA_ASSETS_BASE_DIR  # noqa: E402


def main(params_path: str, task_index: int, episodes: list[int], num_batches: int, batch_size: int, num_workers: int):
    cfg = _config.get_config("pi_behavior_b1k_fast_100task")
    cfg = dataclasses.replace(
        cfg,
        exp_name="eval_tmp",
        assets_base_dir=ILIA_ASSETS_BASE_DIR,
        batch_size=batch_size,
        num_workers=num_workers,
        wandb_enabled=False,
        seed=0,
        weight_loader=weight_loaders.PiBehaviorWeightLoader(params_path),
        model=dataclasses.replace(cfg.model, use_correlated_noise=True),
        data=dataclasses.replace(
            cfg.data,
            base_config=dataclasses.replace(cfg.data.base_config, episodes_index=episodes),
        ),
    )
    mesh = sharding.make_mesh(cfg.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    loader = _data_loader.create_behavior_data_loader(cfg, sharding=data_sharding, shuffle=True)
    norm_stats = loader.data_config().norm_stats
    state, _ = _train.init_train_state(cfg, jax.random.key(0), mesh, resume=False, norm_stats=norm_stats)
    model = nnx.merge(state.model_def, state.params)
    graphdef, params = nnx.split(model)
    del state

    @jax.jit
    def loss_metrics(params, rng, obs, actions):
        m = nnx.merge(graphdef, params)
        d = m.compute_detailed_loss(rng, obs, actions, train=True, num_flow_samples=cfg.num_flow_samples)
        return {k: jnp.mean(v) for k, v in d.items() if hasattr(v, "ndim")}

    sums, n = {}, 0
    it = iter(loader)
    for b in range(num_batches):
        obs, actions = next(it)
        with sharding.set_mesh(mesh):
            out = jax.device_get(loss_metrics(params, jax.random.fold_in(jax.random.key(1234), b), obs, actions))
        for k, v in out.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        n += 1
        print(f"batch {b+1}/{num_batches} loss={float(out['total_loss']):.4f}", flush=True)
    res = {k: v / n for k, v in sorted(sums.items())}
    res["loss"] = res["total_loss"]  # the name the training log prints
    res.update(task_index=task_index, episodes=episodes, num_batches=num_batches, batch_size=batch_size)
    print("EVAL_LOSS_JSON:" + json.dumps(res), flush=True)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--params", required=True)
    p.add_argument("--task-index", type=int, required=True)
    p.add_argument("--episodes", type=int, nargs="+", required=True)
    p.add_argument("--num-batches", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=16)
    a = p.parse_args()
    main(a.params, a.task_index, a.episodes, a.num_batches, a.batch_size, a.num_workers)
