"""B1K Policy Transforms

Transforms BEHAVIOR-1K observations to model format.

Reference: https://github.com/wensi-ai/openpi/blob/behavior/src/openpi/policies/b1k_policy.py
"""

import dataclasses
from collections import OrderedDict

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# Copied from omnigibson.eval.utils.eval_utils.PROPRIOCEPTION_INDICES["R1Pro"]
# (BEHAVIOR-1K/OmniGibson/omnigibson/eval/utils/eval_utils.py, 2026 codebase,
# verified 2026-09-06) instead of importing it. `import omnigibson` pulls in
# omnigibson.envs -> ... -> a native (USD/Isaac-Sim-adjacent) extension that
# segfaults on import in this environment with no Python-level traceback --
# bisected down to omnigibson.envs.data_wrapper/env_base but not further
# (no gdb/strace available here). Training is pure JAX over pre-recorded
# LeRobot data and never touches the simulator, so pulling in that entire
# chain for a 15-line index dict isn't worth the crash risk. If this drifts
# from upstream, training will silently use stale indices -- re-diff against
# the source above if state-dependent behavior looks wrong.
PROPRIOCEPTION_INDICES = {
    "R1Pro": OrderedDict(
        {
            "base_qvel": np.s_[0:3],
            "arm_left_qpos": np.s_[3:10],
            "arm_left_qvel": np.s_[10:17],
            "eef_left_pos": np.s_[17:20],
            "eef_left_quat": np.s_[20:24],
            "gripper_left_qpos": np.s_[24:26],
            "gripper_left_qvel": np.s_[26:28],
            "arm_right_qpos": np.s_[28:35],
            "arm_right_qvel": np.s_[35:42],
            "eef_right_pos": np.s_[42:45],
            "eef_right_quat": np.s_[45:49],
            "gripper_right_qpos": np.s_[49:51],
            "gripper_right_qvel": np.s_[51:53],
            "trunk_qpos": np.s_[53:57],
            "trunk_qvel": np.s_[57:61],
        }
    ),
}


# The pinned (2025-era) omnigibson.learning eval client sends the 256-dim proprio layout
# (copied from BEHAVIOR-1K/OmniGibson/omnigibson/learning/utils/eval_utils.py, only the
# fields extract_state_from_proprio needs). The 2026 training data uses the 61-dim layout
# above, so applying the 61-dim offsets to a 256-dim eval observation reads unrelated
# entries (wrong joints, etc.) and feeds garbage state to the model.
PROPRIOCEPTION_INDICES_256 = {
    "R1Pro": OrderedDict(
        {
            "arm_left_qpos": np.s_[158:165],
            "gripper_left_qpos": np.s_[193:195],
            "arm_right_qpos": np.s_[197:204],
            "gripper_right_qpos": np.s_[232:234],
            "trunk_qpos": np.s_[236:240],
            "base_qpos": np.s_[244:247],
            "base_qvel": np.s_[253:256],
        }
    ),
}


def _proprio_indices(proprio_data):
    dim = np.shape(proprio_data)[-1]
    if dim == 61:
        return PROPRIOCEPTION_INDICES["R1Pro"]
    if dim == 256:
        return PROPRIOCEPTION_INDICES_256["R1Pro"]
    raise ValueError(f"Unsupported proprio dimension {dim}; expected 61 (2026 layout) or 256 (eval client layout)")


def make_b1k_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/egocentric_camera": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_right": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(23),
        "prompt": "do something",
    }

def extract_state_from_proprio(proprio_data):
    """
    We assume perfect correlation for the two gripper fingers.
    """
    idx = _proprio_indices(proprio_data)
    # extract joint position
    base_qvel = proprio_data[..., idx["base_qvel"]]  # 3
    if "base_qpos" in idx:
        # Eval client layout: base_qvel is the WORLD-frame velocity of the virtual x/y/yaw base joints, while the
        # 2026 training state (61-d layout) stores it in the ROBOT frame. Rotate (vx, vy) by -yaw to match training.
        yaw = proprio_data[..., idx["base_qpos"]][..., 2:3]
        c, s = np.cos(yaw), np.sin(yaw)
        vx, vy, wz = base_qvel[..., 0:1], base_qvel[..., 1:2], base_qvel[..., 2:3]
        base_qvel = np.concatenate([c * vx + s * vy, -s * vx + c * vy, wz], axis=-1)
    trunk_qpos = proprio_data[..., idx["trunk_qpos"]]  # 4
    arm_left_qpos = proprio_data[..., idx["arm_left_qpos"]]  #  7
    arm_right_qpos = proprio_data[..., idx["arm_right_qpos"]]  #  7
    
    # Extract raw gripper widths and normalize to [-1, 1] to match action space
    left_gripper_raw = proprio_data[..., idx["gripper_left_qpos"]].sum(axis=-1, keepdims=True)
    right_gripper_raw = proprio_data[..., idx["gripper_right_qpos"]].sum(axis=-1, keepdims=True)
    
    # Normalize gripper widths from [0, 0.1] to [-1, 1] 
    # Based on statistics: physical range is [0, 0.1], action range is [-1, 1]
    # Formula: normalized = 2 * (raw / max_width) - 1
    MAX_GRIPPER_WIDTH = 0.1  # From statistics q99 values
    left_gripper_width = 2.0 * (left_gripper_raw / MAX_GRIPPER_WIDTH) - 1.0
    right_gripper_width = 2.0 * (right_gripper_raw / MAX_GRIPPER_WIDTH) - 1.0

    # Original baseline uses incorrect order for the state
    return np.concatenate([
        base_qvel,
        trunk_qpos,
        arm_left_qpos,
        left_gripper_width,    # Now normalized [-1, 1]
        arm_right_qpos,
        right_gripper_width,   # Now normalized [-1, 1]
    ], axis=-1)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class B1kInputs(transforms.DataTransformFn):
    # Determines which model will be used (not actually used in B1K, kept for compatibility)
    model_type: _model.ModelType | str = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:

        proprio_data = data["observation/state"]
        # extract joint position
        state = extract_state_from_proprio(proprio_data)
        if "actions" in data:
            action =  data["actions"]

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        base_image = _parse_image(data["observation/egocentric_camera"])
        wrist_image_left = _parse_image(data["observation/wrist_image_left"])
        wrist_image_right = _parse_image(data["observation/wrist_image_right"])

        # For B1K, always use 3 cameras (base, left_wrist, right_wrist)
        names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        images = (base_image, wrist_image_left, wrist_image_right)
        image_masks = (np.True_, np.True_, np.True_)

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = action

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
            
        # Preserve task_index for PI_BEHAVIOR model
        if "task_index" in data:
            inputs["task_index"] = data["task_index"]
            
        # Preserve tokenized_prompt for PI_BEHAVIOR model
        if "tokenized_prompt" in data:
            inputs["tokenized_prompt"] = data["tokenized_prompt"]
        if "tokenized_prompt_mask" in data:
            inputs["tokenized_prompt_mask"] = data["tokenized_prompt_mask"]
            
        # Preserve subtask_state for PI_BEHAVIOR model
        if "subtask_state" in data:
            inputs["subtask_state"] = data["subtask_state"]
            
        # Preserve timestamp and episode_index for subtask state computation
        if "timestamp" in data:
            inputs["timestamp"] = data["timestamp"]
        if "episode_index" in data:
            inputs["episode_index"] = data["episode_index"]
            
        # Preserve initial_actions for inpainting
        if "initial_actions" in data:
            initial_actions = data["initial_actions"]
            # Pad initial_actions from 23 dimensions to 32 dimensions (model's action_dim)
            if initial_actions.shape[-1] < 32:
                padding_dim = 32 - initial_actions.shape[-1]
                padding = np.zeros(initial_actions.shape[:-1] + (padding_dim,))
                initial_actions = np.concatenate([initial_actions, padding], axis=-1)
            inputs["initial_actions"] = initial_actions

        return inputs


@dataclasses.dataclass(frozen=True)
class B1kOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Return actions (truncated to 23 dims) and preserve subtask predictions
        result = {"actions": np.asarray(data["actions"][:, :23])}
        
        # Preserve subtask prediction fields for PI_BEHAVIOR models
        if "subtask_logits" in data:
            result["subtask_logits"] = data["subtask_logits"]
        if "predicted_stage" in data:
            result["predicted_stage"] = data["predicted_stage"]
            
        return result
