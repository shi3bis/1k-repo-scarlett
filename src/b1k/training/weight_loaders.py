"""Weight loaders for PI_BEHAVIOR model initialization from Pi05 checkpoints.

Reference: https://github.com/Physical-Intelligence
"""

import dataclasses
import logging
import re

import flax.traverse_util
import jax
import numpy as np
import orbax.checkpoint as ocp

import openpi.shared.array_typing as at
import openpi.shared.download as download

# Re-export base loaders from OpenPI
from openpi.training.weight_loaders import (
    WeightLoader,
    NoOpWeightLoader,
    CheckpointWeightLoader,
    _merge_params,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class PiBehaviorWeightLoader(WeightLoader):
    """Loads checkpoints for PI_BEHAVIOR model.
    
    Automatically detects:
    - Pi05 checkpoint: Loads weights, preserves new PI_BEHAVIOR parameters
    - PI_BEHAVIOR checkpoint: Loads all weights directly
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # Load checkpoint
        params_path = download.maybe_download(self.params_path)
        
        # Load directly with PyTreeCheckpointer (handles both old and new checkpoint formats)
        # No restore target is passed: on disk every leaf is nnx.State-shaped
        # ({"value": array}, one level deeper than a bare flax params dict),
        # nested under "params" -- trying to hand orbax a same-shaped item
        # built from `params` (already flattened/unwrapped) never matches,
        # since reproducing the exact on-disk nesting by hand is fragile and
        # this loader has to work for both Pi05 and PI_BEHAVIOR checkpoints.
        # Instead we let orbax infer structure from its own on-disk metadata,
        # and override ONLY the sharding on every leaf: without that
        # override, orbax defaults to whatever device the checkpoint was
        # *saved* on, and this checkpoint's embedding-table extension was
        # done as plain array surgery on CPU (no GPU needed for that), so its
        # saved sharding references a "cpu:0" device that plain
        # jax.local_devices() (GPU-only view, in this process) doesn't
        # enumerate -- restore fails with "Device cpu:0 was not found in
        # jax.local_devices()" otherwise.
        single_device_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        with ocp.PyTreeCheckpointer() as ckptr:
            metadata = ckptr.metadata(params_path)
            restore_args = jax.tree_util.tree_map(
                lambda m: ocp.ArrayRestoreArgs(sharding=single_device_sharding, dtype=m.dtype),
                metadata,
                is_leaf=lambda m: isinstance(m, ocp.metadata.ArrayMetadata),
            )
            restored = ckptr.restore(params_path, restore_args=restore_args)
        
        # Handle nested 'params' key (from some checkpoint formats)
        if isinstance(restored, dict) and "params" in restored:
            loaded_params = restored["params"]
        else:
            loaded_params = restored
        
        # Remove 'value' suffixes (from nnx.State format)
        flat_params = flax.traverse_util.flatten_dict(loaded_params)
        if all(kp[-1] == "value" for kp in flat_params if len(kp) > 0):
            flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
            loaded_params = flax.traverse_util.unflatten_dict(flat_params)
        
        # Detect checkpoint type
        has_task_embeddings = 'task_embeddings' in loaded_params
        
        if has_task_embeddings:
            # Loading PI_BEHAVIOR checkpoint - load ALL weights from checkpoint
            logging.info("Loading PI_BEHAVIOR checkpoint (all weights will be loaded)")
            # Use _merge_params with empty missing_regex to validate shapes
            return _merge_params(loaded_params, params, missing_regex="^$")
        else:
            # Loading Pi05 checkpoint - preserve new PI_BEHAVIOR-specific parameters
            logging.info("Loading Pi05 checkpoint (new PI_BEHAVIOR parameters will use random init)")
            
            # These parameters are NEW in PI_BEHAVIOR (not in Pi05), so keep them from params (random init)
            missing_regex = (
                ".*task_embeddings.*|"
                ".*stage_pred_from_vlm.*|"
                ".*task_stage_embeddings.*|"
                ".*gate_sincos.*|"
                ".*gate_task_stage.*|"
                ".*gate_task.*|"
                ".*fusion_layer.*|"
                ".*stage_projection.*|"
                ".*task_subtask_fusion.*|"
                ".*fast_token_embedding.*|"
                ".*fast_token_proj.*|"
                ".*kv_transform.*"
            )
            return _merge_params(loaded_params, params, missing_regex=missing_regex)
