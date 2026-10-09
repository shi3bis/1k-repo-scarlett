"""Data loading for BEHAVIOR-1K dataset.

Reference: https://github.com/wensi-ai/openpi/tree/behavior
"""

import json
import logging
import time
from pathlib import Path

# Import all base data loading from OpenPI
from openpi.training.data_loader import (
    Dataset,
    IterableDataset,
    DataLoader,
    TransformedDataset,
    IterableTransformedDataset,
    FakeDataset,
    TorchDataLoader,
    RLDSDataLoader,
    create_torch_dataset,
    create_rlds_dataset,
    transform_iterable_dataset,
    create_data_loader,
    create_torch_data_loader,
    create_rlds_data_loader,
)

import openpi.training.config as _config
import openpi.transforms as _transforms

from b1k.models.observation import Observation
from b1k.transforms_normalize import NormalizeWithPerTimestamp


class DataLoaderImpl(DataLoader):
    """Custom DataLoader using our Observation with fast_tokens."""
    
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield Observation.from_dict(batch), batch["actions"]


def _drop_depth_features(root: Path) -> None:
    """Remove the depth_linear video features from the local meta/info.json.

    The policy only uses the rgb cameras, but LeRobotDataset downloads (and
    decodes every sample) all video features listed in info.json. For the
    joint 76+80+87 run the depth videos alone filled the overlay disk
    (2026-10-08). Editing the local file (original kept as info.json.orig)
    instead of monkeypatching lerobot means the dataset object, and the
    pickled copies dataloader workers get, never see the depth features.
    """
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return
    info = json.loads(info_path.read_text())
    depth = [k for k in info.get("features", {}) if "depth" in k]
    if not depth:
        return
    backup = info_path.with_name("info.json.orig")
    if not backup.exists():
        backup.write_text(info_path.read_text())
    for k in depth:
        del info["features"][k]
    info_path.write_text(json.dumps(info, indent=4))
    logging.info(f"Dropped depth features from {info_path}: {depth}")


def create_behavior_dataset(data_config: _config.DataConfig, action_horizon: int, seed: int | None = None) -> Dataset:
    """Create a BEHAVIOR-1K dataset for training.

    Uses the stock lerobot.datasets.lerobot_dataset.LeRobotDataset, not
    omnigibson.learning.datas.lerobot_dataset.BehaviorLeRobotDataset -- the
    latter no longer exists in the installed 2026 omnigibson package (that
    package renamed/restructured `omnigibson.learning` -> `omnigibson.eval`
    and dropped this class; see the PROPRIOCEPTION_INDICES comment in
    b1k_policy.py for the same root cause, and note importing the full
    `omnigibson` package also segfaults in this environment regardless, in
    unrelated native code -- unnecessary anyway since training never touches
    the simulator). Task filtering is done via data_config.episodes_index
    (an explicit episode index list) instead of the old task-name list,
    since this dataset class has no notion of BEHAVIOR-1K task names.

    video_backend is forced to "pyav": torchcodec is importable here but its
    native library fails to load (ffmpeg .so version mismatch in this
    environment -- lerobot's backend auto-detection only checks whether the
    torchcodec package is findable, not whether it actually loads, so it
    picks torchcodec and crashes at decode time unless overridden).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Use random seed if not provided
    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
        logging.info(f"Using random seed for LeRobotDataset: {seed}")

    _drop_depth_features(Path(data_config.behavior_dataset_root))
    dataset = LeRobotDataset(
        data_config.repo_id,
        root=data_config.behavior_dataset_root,
        delta_timestamps={
            key: [t / 30.0 for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        episodes=data_config.episodes_index,
        video_backend="pyav",
        # See openpi.training.data_loader.create_torch_dataset's identical
        # override: default 1e-4s tolerance is tighter than float32
        # timestamp rounding at these magnitudes.
        tolerance_s=1 / 60,
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset.meta.tasks)])

    return dataset


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform dataset with B1K-specific per-timestamp normalization support.
    
    CRITICAL: This overrides wensi-ai's transform_dataset to pass use_per_timestamp to Normalize.
    wensi-ai's version doesn't support per-timestamp normalization which causes huge action losses!
    """
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    # Build transform list
    transforms_list = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        # Use custom Normalize with per-timestamp support (wensi-ai's doesn't have it!)
        NormalizeWithPerTimestamp(
            norm_stats, 
            use_quantiles=data_config.use_quantile_norm,
            use_per_timestamp=data_config.use_per_timestamp_norm  # CRITICAL: Per-timestamp normalization!
        ),
    ]
    
    # Add subtask state computation for PI_BEHAVIOR models (needs dataset reference)
    model_transforms = []
    for transform in data_config.model_transforms.inputs:
        # ComputeSubtaskStateFromMeta needs dataset reference to access episode lengths
        if hasattr(transform, '__class__') and transform.__class__.__name__ == 'ComputeSubtaskStateFromMeta':
            # Replace placeholder with dataset-aware version
            from b1k import transforms as b1k_transforms
            if hasattr(dataset, 'meta') and hasattr(dataset.meta, 'episodes'):
                model_transforms.append(b1k_transforms.ComputeSubtaskStateFromMeta(dataset=dataset))
                logging.info("Added dataset-aware ComputeSubtaskStateFromMeta transform")
            else:
                logging.warning("Skipping subtask state computation - dataset has no meta.episodes")
        else:
            model_transforms.append(transform)
    
    transforms_list.extend(model_transforms)

    return TransformedDataset(dataset, transforms_list)


def extract_episode_lengths_from_dataset(dataset) -> dict[int, float]:
    """Extract episode lengths from B1K dataset metadata.
    
    Args:
        dataset: BehaviorLeRobotDataset instance
        
    Returns:
        Dictionary mapping episode_index to episode_length (in frames)
        
    Raises:
        ValueError: If dataset doesn't have required metadata
    """
    if not hasattr(dataset, 'episode_data_index'):
        raise ValueError("Dataset must have episode_data_index attribute")
    
    episode_data_index = dataset.episode_data_index
    if 'to' not in episode_data_index or 'from' not in episode_data_index:
        raise ValueError("episode_data_index must have 'to' and 'from' keys")
    
    episode_to = episode_data_index['to'] 
    episode_from = episode_data_index['from']
    episodes = dataset.episodes
    
    episode_lengths = {}
    for i, episode_index in enumerate(episodes):
        if i < len(episode_to) and i < len(episode_from):
            episode_length = episode_to[i] - episode_from[i]
            episode_lengths[episode_index] = float(episode_length)
    
    logging.info(f"Extracted {len(episode_lengths)} episode lengths from dataset")
    return episode_lengths


def create_behavior_data_loader(
    config: _config.TrainConfig,
    *,
    sharding=None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
) -> DataLoader:
    """Create a data loader for BEHAVIOR-1K training."""
    import jax
    import time
    
    data_config = config.data.create(config.assets_dirs, config.model)
    
    # Use random seed if not provided
    seed = config.seed
    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
        logging.info(f"Using random seed: {seed}")
    
    dataset = create_behavior_dataset(data_config, action_horizon=config.model.action_horizon, seed=seed)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=seed,
    )
    
    return DataLoaderImpl(data_config, data_loader)
