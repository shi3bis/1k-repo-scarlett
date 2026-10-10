"""Build a base checkpoint whose task_stage_embeddings table matches the
current TASK_NUM_STAGES (pi_behavior_config.py), starting from ext100_t80t87_t63t78.

ext100_t80t87_t63t78's table is laid out for OLD_TASK_NUM_STAGES (the 656-row layout of
branch attach-35-63-78: 2025 tasks 0-49 + 55/63/76/78/80/82/87). This branch adds the
BDDL `covered` tasks 57 (5 stages), 60 (4), 69 (3), 93 (5), ceil(mean episode length / 900).
Rows are indexed by cumulative offsets in task-index order, so a new task can't
just be appended at the end: its rows must be INSERTED at its offset, shifting
every later task. This rebuilds the table task by task: existing tasks' rows are
copied byte-identically, new tasks get rows drawn from N(mean, std) of the
existing table (same init extend_embeddings.py used for 55/76/82).
task_embeddings already has rows for all 100 tasks.

--src/--dst also let it re-lay-out any other 656-row checkpoint the same way.

Run on CPU so it doesn't contend for the GPU:
  JAX_PLATFORMS=cpu PYTHONPATH=<worktree>/src python scripts/extend_stage_embeddings.py
"""
import sys
from pathlib import Path

import numpy as np
import orbax.checkpoint as ocp
from flax import traverse_util
from openpi.models import model as _model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from b1k.models.pi_behavior_config import TASK_NUM_STAGES  # noqa: E402

SRC = "/workspace/persistent/behavior/checkpoints/behavior_50t_checkpoint_ext100_t80t87_t63t78/params"
DST = "/workspace/persistent/behavior/checkpoints/behavior_50t_checkpoint_ext100_t80t87_t63t78_cov4/params"
SEED = 0

# The layout ext100_t80t87_t63t78 was built with: 2025 tasks 0-49 + 55/63/76/78/80/82/87.
OLD_TASK_NUM_STAGES = list(TASK_NUM_STAGES)
for _t in (57, 60, 69, 93):
    OLD_TASK_NUM_STAGES[_t] = 0
assert sum(OLD_TASK_NUM_STAGES) == 656

def restore(path):
    # openpi's own loader (the version-specific orbax metadata API differs from
    # what extend_embeddings.py was written against).
    return _model.restore_params(path, restore_type=np.ndarray)


def offsets(counts):
    return [0] + list(np.cumsum(counts)[:-1])


def main(src=SRC, dst=DST):
    params = restore(src)
    tse = np.asarray(params["task_stage_embeddings"]["embedding"])
    assert tse.shape[0] == sum(OLD_TASK_NUM_STAGES), f"{tse.shape} doesn't match OLD layout ({sum(OLD_TASK_NUM_STAGES)})"

    rng = np.random.default_rng(SEED)
    mean, std = tse.mean(), tse.std()
    old_off, new_off = offsets(OLD_TASK_NUM_STAGES), offsets(TASK_NUM_STAGES)
    blocks, added = [], []
    for t, n in enumerate(TASK_NUM_STAGES):
        if n == 0:
            continue
        if OLD_TASK_NUM_STAGES[t] == n:
            blocks.append(tse[old_off[t]:old_off[t] + n])
        elif OLD_TASK_NUM_STAGES[t] == 0:
            blocks.append(rng.normal(loc=mean, scale=std, size=(n, tse.shape[1])).astype(tse.dtype))
            added.append((t, n, new_off[t]))
        else:
            raise ValueError(f"task {t}: stage count changed {OLD_TASK_NUM_STAGES[t]} -> {n}; not supported")
    tse_new = np.concatenate(blocks, axis=0)
    assert tse_new.shape[0] == sum(TASK_NUM_STAGES)

    # Every pre-existing task's rows must be unchanged at its NEW offset.
    for t, n in enumerate(OLD_TASK_NUM_STAGES):
        if n:
            assert np.array_equal(tse_new[new_off[t]:new_off[t] + n], tse[old_off[t]:old_off[t] + n]), t
    print(f"task_stage_embeddings {tse.shape} -> {tse_new.shape}; inserted (task, rows, offset): {added}")

    params["task_stage_embeddings"]["embedding"] = tse_new
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    # Same {"params": ...} layout with nnx.State's trailing "value" keys that
    # openpi's save_state writes (restore_params strips them on load).
    flat = traverse_util.flatten_dict(params)
    to_save = traverse_util.unflatten_dict({(*kp, "value"): v for kp, v in flat.items()})
    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(dst, {"params": to_save})

    rparams = restore(dst)
    assert np.array_equal(np.asarray(rparams["task_stage_embeddings"]["embedding"]), tse_new)
    assert np.array_equal(np.asarray(rparams["task_embeddings"]["embedding"]), np.asarray(params["task_embeddings"]["embedding"]))
    print("saved + verified:", dst)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--src", default=SRC)
    p.add_argument("--dst", default=DST)
    a = p.parse_args()
    main(a.src, a.dst)
