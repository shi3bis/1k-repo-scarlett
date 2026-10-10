"""Custom eval driver for the 4 BEHAVIOR Challenge 2026 tasks this repo targets
(picking_up_trash=1, re_shelving_library_books=55, dispose_of_glass=76,
store_batteries=82), against the pinned BEHAVIOR-1K submodule commit's eval
harness (omnigibson.learning.eval), which predates the 2026 task additions.

Why not just call `omnigibson/learning/eval.py` directly: its normal
instance-resolution path reads `2025-challenge-task-instances/metadata/
test_instances.csv`, which only has rows for the original 50 tasks (0-49) --
IndexErrors for 55/76/82. Its `TASK_NAMES_TO_INDICES` dict is also missing
those 3 tasks, and the bundled `joylo/sampled_task/available_tasks.yaml` (used
to find each task's scene_model + robot start pose) is missing 2 of them
(dispose_of_glass, store_batteries) entirely.

This script reuses the Evaluator class and Hydra config plumbing unmodified
(stays wire-compatible with scripts/serve_b1k.py's websocket server), but:
  - monkeypatches TASK_NAMES_TO_INDICES with the 3 missing/correct task ids
    (matching what scripts/joint_loop.py actually trained the checkpoint
    with), confirmed against 2026-challenge-task-instances/metadata/
    B100_task_misc.csv's "Task ID" column.
  - sources scene_model / robot_start_position / robot_start_orientation from
    2026-challenge-task-instances/metadata/available_tasks.yaml (has all 4
    tasks) instead of the stale bundled YAML (has 2 of them).
  - points task-instance TRO-state loading at the 2026-challenge-task-instances
    dataset instead of the hardcoded 2025 one.
  - skips the episodes.jsonl-based human-stats/max-steps lookup (not present
    for new tasks) in favor of an explicit --max-steps.
  - takes literal instance ids instead of resolving through the stale
    2025-only test_instances.csv.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
# torch.compile/inductor triggers a fresh C++ JIT compile per process (irrelevant
# for a single slow eval rollout) and writes its temp build artifacts under
# TMPDIR -- on this host that's cache/tmp on the same quota-constrained volume
# as everything else, and it was observed tipping a run over the edge with
# "fatal error: error writing to .../ccXXXXXX.s: Disk quota exceeded". Disabling
# it falls back to plain eager-mode torch, which is what we want here anyway.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import hydra
import yaml
from omegaconf import OmegaConf

# --- monkeypatches must happen before Evaluator's module-level state is used ---
import omnigibson.learning.utils.eval_utils as eval_utils_mod

NEW_TASK_INDICES = {
    "re_shelving_library_books": 55,
    "dispose_of_glass": 76,
    "store_batteries": 82,
    # 2026-10-04: chain-76-80-87 tasks, ids from 2026-challenge-task-instances/metadata/B100_task_misc.csv
    "clean_up_broken_glass": 80,
    "collecting_aluminum_cans": 87,
    # 2026-10-09: attached-predicate tasks (35 is in the original 50 already)
    "installing_smoke_detectors": 63,
    "make_cabinet_doors": 78,
}
eval_utils_mod.TASK_NAMES_TO_INDICES.update(NEW_TASK_INDICES)
eval_utils_mod.TASK_INDICES_TO_NAMES = {v: k for k, v in eval_utils_mod.TASK_NAMES_TO_INDICES.items()}

# 2026-10-04: asset-version drift. The 2026 task-instance JSONs were saved with newer object assets than our
# behavior-1k-assets 3.7.2rc1; e.g. restaurant_diner (task 80) has 4x metal_bottom_cabinet_bbjuqu saved with 2 joint
# positions while the local bbjuqu USD has 1 joint (door) -> "shape mismatch ... [2] ... [1, 1]" in scene init.
# Fit saved joint_pos/joint_vel to the loaded object's DOF (truncate / zero-pad) and log it, instead of crashing.
import omnigibson.prims.entity_prim as _entity_prim_mod

_orig_entity_load_state = _entity_prim_mod.EntityPrim._load_state


def _dof_tolerant_load_state(self, state):
    if "joints" not in state and self.n_joints > 0 and "joint_pos" in state and len(state["joint_pos"]) != self.n_dof:
        import torch as _th

        def _fit(x):
            x = _th.as_tensor(x, dtype=_th.float32).flatten()
            return x[: self.n_dof] if len(x) >= self.n_dof else _th.cat([x, _th.zeros(self.n_dof - len(x))])

        print(f"[run_custom_eval] DOF mismatch for {self.name}: saved {len(state['joint_pos'])} joint values, "
              f"object has {self.n_dof} -> fitted", flush=True)
        state = dict(state, joint_pos=_fit(state["joint_pos"]), joint_vel=_fit(state["joint_vel"]))
    return _orig_entity_load_state(self, state)


_entity_prim_mod.EntityPrim._load_state = _dof_tolerant_load_state

DATASET_2026_ROOT = None  # set in main() once gm.DATA_PATH is known

# The pinned bddl package's activity_definitions/<task>/problem0.bddl for these 3 tasks
# predate the 2026 re-sampling (object counts/types changed -- e.g. store_batteries used
# to have 2 batteries + a shelf, the 2026 template has 3 batteries + 2 cabinets + a
# countertop + a coffee_table, no shelf at all) -- loading the stale local file makes
# BehaviorTask.assign_object_scope_with_cache() assert on synsets that don't exist in the
# actually-sampled scene. picking_up_trash (task 1, in the original 50) is untouched --
# its local problem0.bddl already matches. Fetched from StanfordVL/BEHAVIOR-1K main's
# bddl3/bddl/activity_definitions/<task>/problem0.bddl (the current, 2026-era package;
# not used directly since it's a different package/domain layout than this pinned
# "bddl" -- just borrowing its up-to-date problem text) and passed in via
# cfg["task"]["predefined_problem"], which BehaviorTask accepts as a raw BDDL string,
# rather than editing the vendored file in the submodule checkout.
TASK_BDDL_OVERRIDES = {
    "re_shelving_library_books": """\
(define (problem re_shelving_library_books-0)
    (:domain omnigibson)

    (:objects
     	book.n.02_1 book.n.02_2 book.n.02_3 - book.n.02
    	table.n.02_1 - table.n.02
    	bookcase.n.01_1 bookcase.n.01_2 bookcase.n.01_3 bookcase.n.01_4 - bookcase.n.01
    	floor.n.01_1 - floor.n.01
    	agent.n.01_1 - agent.n.01
    )

    (:init
        (ontop book.n.02_1 table.n.02_1)
        (ontop book.n.02_2 table.n.02_1)
        (ontop book.n.02_3 table.n.02_1)
        (inroom table.n.02_1 living_room)
        (inroom bookcase.n.01_1 living_room)
        (inroom bookcase.n.01_2 living_room)
        (inroom bookcase.n.01_3 living_room)
        (inroom bookcase.n.01_4 living_room)
        (inroom floor.n.01_1 living_room)
        (ontop agent.n.01_1 floor.n.01_1)
    )

    (:goal
        (and
            (forall
                (?book.n.02 - book.n.02)
                (exists
                    (?bookcase.n.01 - bookcase.n.01)
                    (inside ?book.n.02 ?bookcase.n.01)
                )
            )
        )
    )
)
""",
    "dispose_of_glass": """\
(define (problem dispose_of_glass-0)
    (:domain omnigibson)

    (:objects
        water_glass.n.02_1 water_glass.n.02_2 water_glass.n.02_3 water_glass.n.02_4 - water_glass.n.02
        ashcan.n.01_1 - ashcan.n.01
        floor.n.01_1 floor.n.01_2 - floor.n.01
        sink.n.01_1 - sink.n.01
        agent.n.01_1 - agent.n.01
    )

    (:init
        (inroom sink.n.01_1 bathroom)
        (inroom floor.n.01_1 bathroom)
        (inroom floor.n.01_2 bedroom)
        (ontop water_glass.n.02_1 sink.n.01_1)
        (ontop water_glass.n.02_2 sink.n.01_1)
        (ontop water_glass.n.02_3 sink.n.01_1)
        (ontop water_glass.n.02_4 sink.n.01_1)
        (ontop ashcan.n.01_1 floor.n.01_2)
        (ontop agent.n.01_1 floor.n.01_1)
    )

    (:goal
        (and
            (forall
                (?water_glass.n.02 - water_glass.n.02)
                (inside ?water_glass.n.02 ?ashcan.n.01_1)
            )
        )
    )
)
""",
    "store_batteries": """\
(define (problem store_batteries-0)
    (:domain omnigibson)

    (:objects
        battery.n.02_1 battery.n.02_2 battery.n.02_3 - battery.n.02
        coffee_table.n.01_1 - coffee_table.n.01
        cabinet.n.01_1 cabinet.n.01_2 - cabinet.n.01
        countertop.n.01_1 - countertop.n.01
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )

    (:init
        (ontop battery.n.02_1 coffee_table.n.01_1)
        (ontop battery.n.02_2 coffee_table.n.01_1)
        (ontop battery.n.02_3 countertop.n.01_1)
        (inroom coffee_table.n.01_1 living_room)
        (inroom cabinet.n.01_1 living_room)
        (inroom cabinet.n.01_2 living_room)
        (inroom countertop.n.01_1 kitchen)
        (inroom floor.n.01_1 living_room)
        (ontop agent.n.01_1 floor.n.01_1)
    )

    (:goal
        (and
            (exists
                (?cabinet.n.01 - cabinet.n.01)
                (forall
                    (?battery.n.02 - battery.n.02)
                    (inside ?battery.n.02 ?cabinet.n.01)
                )
            )
        )
    )
)
""",
    # 2026-10-04: task 80. Neither the pinned nor bddl3 problem0 matches the 2026 restaurant_diner template (both have
    # a broom.n.01_1 and one floor; the template's inst_to_name has no broom and floor.n.01_1 = floors_cxcxqq_0
    # (dining_room_0, where the 3 broken_glass are) + floor.n.01_2 = floors_fcxkki_0 (corridor_0, trash_can_96)).
    # Written to match that template (+ metadata/task_custom_lists.json: rooms corridor+dining_room); goal unchanged.
    "clean_up_broken_glass": """\
(define (problem clean_up_broken_glass-0)
    (:domain omnigibson)

    (:objects
        broken__glass.n.01_1 broken__glass.n.01_2 broken__glass.n.01_3 - broken__glass.n.01
        floor.n.01_1 floor.n.01_2 - floor.n.01
        ashcan.n.01_1 - ashcan.n.01
        agent.n.01_1 - agent.n.01
    )

    (:init
        (ontop broken__glass.n.01_1 floor.n.01_1)
        (ontop broken__glass.n.01_2 floor.n.01_1)
        (ontop broken__glass.n.01_3 floor.n.01_1)
        (ontop ashcan.n.01_1 floor.n.01_2)
        (ontop agent.n.01_1 floor.n.01_1)
        (inroom floor.n.01_1 dining_room)
        (inroom floor.n.01_2 corridor)
    )

    (:goal
        (and
            (forall
                (?broken__glass.n.01 - broken__glass.n.01)
                (inside ?broken__glass.n.01 ?ashcan.n.01_1)
            )
        )
    )
)
""",
    # 2026-10-09: task 63. The pinned/bddl3 problem0 has 2 fire alarms + 2 wall nails; the 2026 house_double_floor_lower
    # template has 1 fire_alarm (fire_alarm_102, on coffee_table_koagbh_0 in living_room_0), 1 wall_nail (kitchen_0) and a
    # table.n.02_1. Written to match that template; goal unchanged (the alarm attached to the nail).
    "installing_smoke_detectors": """\
(define (problem installing_smoke_detectors-0)
    (:domain omnigibson)

    (:objects
        wall_nail.n.01_1 - wall_nail.n.01
        floor.n.01_1 - floor.n.01
        table.n.02_1 - table.n.02
        fire_alarm.n.02_1 - fire_alarm.n.02
        agent.n.01_1 - agent.n.01
    )

    (:init
        (ontop fire_alarm.n.02_1 table.n.02_1)
        (inroom wall_nail.n.01_1 kitchen)
        (inroom table.n.02_1 living_room)
        (inroom floor.n.01_1 living_room)
        (ontop agent.n.01_1 floor.n.01_1)
    )

    (:goal
        (and
            (forpairs
                (?fire_alarm.n.02 - fire_alarm.n.02)
                (?wall_nail.n.01 - wall_nail.n.01)
                (attached ?fire_alarm.n.02 ?wall_nail.n.01)
            )
        )
    )
)
""",
}


def _load_available_tasks_2026():
    with open(os.path.join(DATASET_2026_ROOT, "metadata", "available_tasks.yaml")) as f:
        return yaml.safe_load(f)


def _get_task_instance_path_2026(scene_name):
    scenes_path = os.path.join(DATASET_2026_ROOT, "scenes")
    if scene_name in os.listdir(scenes_path):
        return os.path.join(scenes_path, scene_name)
    raise ValueError(f"Could not find scene {scene_name} in {scenes_path}")


import gello.robots.sim_robot.og_teleop_utils as og_teleop_utils_mod  # noqa: E402
import omnigibson as og  # noqa: E402
import omnigibson.learning.eval as eval_mod  # noqa: E402
import omnigibson.utils.asset_utils as asset_utils_mod  # noqa: E402
from gello.robots.sim_robot.og_teleop_cfg import DISABLED_TRANSITION_RULES  # noqa: E402
from gello.robots.sim_robot.og_teleop_utils import (  # noqa: E402
    augment_rooms,
    generate_robot_config,
    get_task_relevant_room_types,
)
from hydra.utils import instantiate  # noqa: E402
from omnigibson.learning.eval import Evaluator  # noqa: E402
from omnigibson.learning.utils.config_utils import register_omegaconf_resolvers  # noqa: E402
from omnigibson.learning.utils.eval_utils import (  # noqa: E402
    PROPRIOCEPTION_INDICES,
    TASK_NAMES_TO_INDICES,
    generate_basic_environment_config,
)
from omnigibson.learning.utils.obs_utils import create_video_writer  # noqa: E402
from omnigibson.macros import gm  # noqa: E402
import omnigibson.utils.transform_utils as T  # noqa: E402
import torch as th  # noqa: E402

logger = logging.getLogger("custom_eval")


def _load_human_stats_2026(task_name: str) -> dict:
    """Replacement for eval.py's inline episodes.jsonl averaging, which reads a
    2025-only path that doesn't exist here. 2026-challenge-task-instances/metadata/
    task.jsonl has the exact same fields (length, distance_traveled,
    left_eef_displacement, right_eef_displacement), pre-aggregated per task -- no
    need to re-average from raw episodes ourselves."""
    task_jsonl = os.path.join(DATASET_2026_ROOT, "metadata", "task.jsonl")
    with open(task_jsonl) as f:
        for line in f:
            row = json.loads(line)
            if row["task_name"] == task_name:
                return {
                    "length": row["length"],
                    "distance_traveled": row["distance_traveled"],
                    "left_eef_displacement": row["left_eef_displacement"],
                    "right_eef_displacement": row["right_eef_displacement"],
                }
    raise ValueError(f"No human-stats row for task {task_name!r} in {task_jsonl}")


def _load_filtered_scene_json(scene_model: str, task_name: str) -> dict:
    """Load the task's own instance-0 template (not the generic <scene>_best.json)
    as the scene_file, and drop any objects_info.init_info entries whose
    class_module references omnigibson.robots.* generically (e.g.
    "omnigibson.robots.robot", which doesn't exist in this pinned OmniGibson
    version -- the newer tooling that generated the 2026 scene data renamed/
    restructured that module). These are robot placeholders left over in the
    scene capture; the actual eval robot is added separately via cfg["robots"],
    and InteractiveTraversableScene._should_load_object() is supposed to filter
    these by class_name against REGISTERED_ROBOTS, but that check misses this
    particular placeholder's class_name, so create_object_from_init_info()
    reaches import_module() on a module path that no longer exists and crashes.
    Filtering here (before the scene even sees them) sidesteps that without
    touching vendored code.

    Using the generic "<scene>_best.json" (rather than the task's own template)
    fails later in BehaviorTask.initialize_activity() -> assign_object_scope_with_cache():
    that needs env.scene.get_task_metadata("inst_to_name"), a BDDL-synset ->
    object-name mapping that's only present in a task-specific scene file's
    metadata.task, not the generic base scene's. The per-task instance-0
    "template" file (already local under 2026-challenge-task-instances) has
    exactly that, plus the same objects as the base scene plus task-relevant
    additions -- it's the correct base state for this task, with the
    per-instance "-tro_state.json" files applying only pose deltas on top via
    load_task_instance()."""
    scene_json_path = os.path.join(
        DATASET_2026_ROOT, "scenes", scene_model, "json", f"{scene_model}_task_{task_name}_0_0_template.json"
    )
    with open(scene_json_path) as f:
        data = json.load(f)
    init_info = data["objects_info"]["init_info"]
    dropped = [name for name, info in init_info.items() if "robots" in info.get("class_module", "")]

    # Also drop objects whose asset simply doesn't exist in our (selectively-extracted,
    # 3.7.2rc1-pinned) local behavior-1k-assets copy -- found one case (a "fixed_window"
    # model referenced by the 2026 dispose_of_glass template) that's absent from BOTH the
    # 3.7.2rc1 zip AND the current upstream latest zip, so it's not something a bigger
    # selective-extraction pass would fix; the model id genuinely doesn't exist anywhere we
    # can reach. These are near-certainly structural/background dressing (windows, etc.),
    # not task-relevant objects (the task's actual BDDL-scoped objects are guaranteed
    # present since we already required them by name in TASK_BDDL_OVERRIDES), so skipping
    # them is safe -- far better than a segfault-on-missing-file crash for something the
    # task doesn't care about. Logged loudly in case it's ever NOT safe for some future task.
    assets_root = asset_utils_mod.get_dataset_path("behavior-1k-assets")
    missing_asset = []
    for name, info in init_info.items():
        if name in dropped:
            continue
        args = info.get("args", {})
        category, model = args.get("category"), args.get("model")
        if category is None or model is None:
            continue
        usd_path = os.path.join(assets_root, "objects", category, model, "usd", f"{model}.encrypted.usd")
        if not os.path.exists(usd_path):
            missing_asset.append(name)
    dropped += missing_asset

    for name in dropped:
        del init_info[name]
    state_registry = (
        data["state"]["registry"]["object_registry"] if "registry" in data["state"] else data["state"]["object_registry"]
    )
    for name in dropped:
        state_registry.pop(name, None)
    if dropped:
        logger.info(f"Filtered {len(dropped)} object(s) out of {scene_model}/{task_name} scene_file: {dropped}")
    if missing_asset:
        logger.warning(f"Skipped {len(missing_asset)} object(s) with no asset on disk (checked {assets_root}): {missing_asset}")
    return data


def _patched_load_task_instance(self, instance_id: int, test_hidden: bool = False) -> None:
    """Copy of Evaluator.load_task_instance with one fix: the 2026 tro_state.json
    files key robot_poses generically as "robot" instead of by robot model_name
    ("R1Pro") -- try the model name first (old-format compatibility) and fall
    back to "robot"."""
    scene_model = self.env.task.scene_name
    tro_filename = self.env.task.get_cached_activity_scene_filename(
        scene_model=scene_model,
        activity_name=self.env.task.activity_name,
        activity_definition_id=self.env.task.activity_definition_id,
        activity_instance_id=instance_id,
    )
    if test_hidden:
        tro_file_path = os.path.join(
            gm.DATA_PATH,
            "2025-challenge-test-instances",
            self.env.task.activity_name,
            f"{tro_filename}-tro_state.json",
        )
    else:
        tro_file_path = os.path.join(
            asset_utils_mod.get_task_instance_path(scene_model),
            f"json/{scene_model}_task_{self.env.task.activity_name}_instances/{tro_filename}-tro_state.json",
        )
    from omnigibson.utils.python_utils import recursively_convert_to_torch

    with open(tro_file_path, "r") as f:
        tro_state = recursively_convert_to_torch(json.load(f))
    for tro_key, tro_val in tro_state.items():
        if tro_key == "robot_poses":
            presampled_robot_poses = tro_val
            robot_key = self.robot.model_name if self.robot.model_name in presampled_robot_poses else "robot"
            robot_pos = presampled_robot_poses[robot_key][0]["position"]
            robot_quat = presampled_robot_poses[robot_key][0]["orientation"]
            self.robot.set_position_orientation(robot_pos, robot_quat)
            self.env.scene.write_task_metadata(key=tro_key, data=tro_val)
        else:
            self.env.task.object_scope[tro_key].load_state(tro_val, serialized=False)

    for _ in range(25):
        og.sim.step_physics()
        for entity in self.env.task.object_scope.values():
            if not entity.is_system and entity.exists:
                entity.keep_still()

    self.env.scene.update_initial_file()
    self.env.scene.reset()


def _patched_load_env(self, env_wrapper):
    """Copy of Evaluator.load_env with the 2025-only episodes.jsonl human-stats
    lookup swapped for _load_human_stats_2026. Everything else unchanged."""
    for rule in DISABLED_TRANSITION_RULES:
        rule.ENABLED = False
    available_tasks = og_teleop_utils_mod.load_available_tasks()
    task_name = self.cfg.task.name
    assert task_name in available_tasks, f"Got invalid task name: {task_name}"
    self.human_stats = _load_human_stats_2026(task_name)

    task_cfg = available_tasks[task_name][0]
    robot_type = self.cfg.robot.type
    assert robot_type == "R1Pro", f"Got invalid robot type: {robot_type}, only R1Pro is supported."
    cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg)
    cfg["scene"]["scene_file"] = _load_filtered_scene_json(task_cfg["scene_model"], task_name)
    if task_name in TASK_BDDL_OVERRIDES:
        cfg["task"]["predefined_problem"] = TASK_BDDL_OVERRIDES[task_name]
    # BehaviorTask defaults use_presampled_robot_pose=True, which requires a
    # "robot_poses" entry in scene task_metadata *before* the first env.reset()
    # (i.e. before load_task_instance() ever runs) -- ours only gets populated
    # per-instance, later. False falls back to the robot's configured starting
    # pose (from task_cfg's robot_start_position/orientation) for that first
    # reset; load_task_instance() still applies the real presampled pose per
    # instance afterwards.
    cfg["task"]["use_presampled_robot_pose"] = False
    if self.cfg.partial_scene_load:
        relevant_rooms = get_task_relevant_room_types(activity_name=task_name)
        relevant_rooms = augment_rooms(relevant_rooms, task_cfg["scene_model"], task_name)
        cfg["scene"]["load_room_types"] = relevant_rooms
    cfg["robots"] = [generate_robot_config(task_name=task_name, task_cfg=task_cfg)]
    cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb"]
    cfg["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())
    if self.cfg.robot.controllers is not None:
        cfg["robots"][0]["controller_config"].update(self.cfg.robot.controllers)
    if self.cfg.max_steps is None:
        logger.info(
            f"Setting timeout to be 2x the average length of human demos: {int(self.human_stats['length'] * 2)}"
        )
        cfg["task"]["termination_config"]["max_steps"] = int(self.human_stats["length"] * 2)
    else:
        logger.info(f"Setting timeout to be {self.cfg.max_steps} steps through config.")
        cfg["task"]["termination_config"]["max_steps"] = self.cfg.max_steps
    cfg["task"]["include_obs"] = False
    env = og.Environment(configs=cfg)
    env = instantiate(env_wrapper, env=env)
    return env
logging.basicConfig(level=logging.INFO, format="%(asctime)s [CUSTOM_EVAL] %(message)s")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task-name", required=True, choices=sorted(TASK_NAMES_TO_INDICES.keys() | NEW_TASK_INDICES.keys()))
    p.add_argument("--instance-ids", type=int, nargs="+", default=[1, 2])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--log-path", default="/workspace/persistent/eval_outputs")
    p.add_argument("--write-video", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--replay-actions", default=None,
                    help="diagnostic: .npy [T,23] of recorded actions replayed open-loop (use with --policy local)")
    p.add_argument("--replay-record", default=None, help="diagnostic: where to save the measured proprio per step (.npy)")
    p.add_argument("--policy", choices=("websocket", "local"), default="websocket",
                    help="local = zero-action smoke test, no policy server needed")
    return p.parse_args()


class ReplayActor:
    """Diagnostic: returns recorded actions step by step and logs the proprio observed BEFORE each action is applied."""

    def __init__(self, actions):
        import numpy as _np
        import torch as _th
        self._np, self._th = _np, _th
        self.actions = _np.asarray(actions, dtype=_np.float32)
        self.t = 0
        self.rec = []

    def act(self, obs):
        pro = obs["robot_r1::proprio"]
        pro = pro.detach().cpu().numpy() if hasattr(pro, "detach") else self._np.asarray(pro)
        self.rec.append(pro.copy())
        a = self.actions[min(self.t, len(self.actions) - 1)]
        self.t += 1
        return self._th.tensor(a, dtype=self._th.float32)

    def reset(self):
        self.t = 0
        self.rec = []


def main():
    global DATASET_2026_ROOT
    args = parse_args()

    register_omegaconf_resolvers()
    DATASET_2026_ROOT = os.path.join(gm.DATA_PATH, "2026-challenge-task-instances")
    assert os.path.isdir(DATASET_2026_ROOT), f"Missing dataset dir: {DATASET_2026_ROOT}"

    # Patch the two lookups eval.py's Evaluator.load_env() uses, in the
    # namespaces it actually reads them from (both eval.py and og_teleop_utils
    # import these by name, so the patch must apply to each importer's binding).
    og_teleop_utils_mod.load_available_tasks = _load_available_tasks_2026
    eval_mod.load_available_tasks = _load_available_tasks_2026
    asset_utils_mod.get_task_instance_path = _get_task_instance_path_2026
    eval_mod.get_task_instance_path = _get_task_instance_path_2026
    import omnigibson.scenes.interactive_traversable_scene as its_mod
    its_mod.get_task_instance_path = _get_task_instance_path_2026
    eval_mod.Evaluator.load_env = _patched_load_env

    # 2026-10-10: also send the robot's WORLD base pose (x, y, yaw) to the policy server. The proprio base_qpos (256-d
    # [244:247]) is the holonomic base's virtual-joint position, measured from the robot's load pose, not world coordinates,
    # so server-side rules that need the room (B1K_ROOM_STAGE_TASK) read this key instead.
    _orig_preprocess_obs = eval_mod.Evaluator._preprocess_obs

    def _preprocess_obs_with_world_pose(self, obs):
        out = _orig_preprocess_obs(self, obs)
        pos, quat = self.robot.get_position_orientation()
        yaw = T.quat2euler(quat)[2]
        out["robot_r1::world_base_pose"] = th.tensor([float(pos[0]), float(pos[1]), float(yaw)], dtype=th.float32)
        return out

    eval_mod.Evaluator._preprocess_obs = _preprocess_obs_with_world_pose
    eval_mod.Evaluator.load_task_instance = _patched_load_task_instance

    eval_config_dir = str(Path(eval_mod.__file__).resolve().parent / "configs")
    overrides = [
        f"task.name={args.task_name}",
        f"policy={args.policy}",
        f"log_path={args.log_path}",
        f"write_video={'true' if args.write_video else 'false'}",
        f"max_steps={args.max_steps}",
        "partial_scene_load=false",
    ]
    if args.policy == "websocket":
        overrides += [f"model.host={args.host}", f"model.port={args.port}"]
    with hydra.initialize_config_dir(eval_config_dir, version_base="1.1"):
        config = hydra.compose("base_config.yaml", overrides=overrides)
    OmegaConf.resolve(config)

    gm.HEADLESS = config.headless

    video_path = None
    if config.write_video:
        video_path = Path(config.log_path).expanduser() / "videos"
        video_path.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(config.log_path).expanduser() / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Task: {args.task_name} (task_id={TASK_NAMES_TO_INDICES[args.task_name]})")
    logger.info(f"Instance ids to run: {args.instance_ids}")

    results = []
    with Evaluator(config) as evaluator:
        replay_actor = None
        if args.replay_actions:
            import numpy as np
            assert args.policy == "local", "--replay-actions requires --policy local"
            replay_actor = ReplayActor(np.load(args.replay_actions))
            evaluator.policy.policy = replay_actor
            logger.info(f"REPLAY MODE: {len(replay_actor.actions)} recorded actions from {args.replay_actions}")
        for idx in args.instance_ids:
            evaluator.reset()
            evaluator.load_task_instance(idx, test_hidden=False)
            logger.info(f"Starting task instance {idx} for evaluation...")
            evaluator.reset()
            done = False
            video_name = None
            if config.write_video:
                video_name = str(video_path / f"{args.task_name}_{idx}_0.mp4")
                evaluator.video_writer = create_video_writer(fpath=video_name, resolution=(448, 672))
            for metric in evaluator.metrics:
                metric.start_callback(evaluator.env)
            steps = 0
            # 2026-10-04: pose log for path analysis (base path + task-object positions). One JSON line per record:
            #   {"t": step, "base": [x, y, yaw]} every 10 steps; {"t": step, "objs": {inst: [x, y, z]}} at step 0 and every
            #   500 steps (task-scope objects only; agent/floor/rooms skipped). File: <log_path>/poses/<task>_<inst>.jsonl
            pose_dir = Path(config.log_path) / "poses"
            pose_dir.mkdir(parents=True, exist_ok=True)
            pose_f = open(pose_dir / f"{args.task_name}_{idx}.jsonl", "w")
            scope = {
                k: v for k, v in (evaluator.env.task.object_scope or {}).items()
                if not k.startswith(("agent.", "floor.")) and getattr(v, "exists", True) and v.wrapped_obj is not None
            }

            def _log_pose(t, with_objs=False):
                pos, quat = evaluator.robot.get_position_orientation()
                yaw = float(T.quat2euler(quat)[2])
                rec = {"t": t, "base": [round(float(pos[0]), 3), round(float(pos[1]), 3), round(yaw, 4)]}
                if with_objs or t % 500 == 0:
                    objs = {}
                    for k, v in scope.items():
                        try:
                            op = v.get_position_orientation()[0]
                            objs[k] = [round(float(op[0]), 3), round(float(op[1]), 3), round(float(op[2]), 3)]
                        except Exception:
                            pass
                    rec["objs"] = objs
                pose_f.write(json.dumps(rec) + "\n")

            _log_pose(0)
            while not done:
                terminated, truncated = evaluator.step()
                steps += 1
                if steps % 10 == 0:
                    _log_pose(steps)
                if terminated or truncated:
                    done = True
                if config.write_video:
                    evaluator._write_video()
                if evaluator.env._current_step % 200 == 0:
                    logger.info(f"instance={idx} step={evaluator.env._current_step}")
            _log_pose(steps, with_objs=True)
            pose_f.close()
            for metric in evaluator.metrics:
                metric.end_callback(evaluator.env)
            if replay_actor is not None and args.replay_record:
                import numpy as np
                np.save(args.replay_record, np.stack(replay_actor.rec))
                logger.info(f"Saved replay proprio record {np.stack(replay_actor.rec).shape} to {args.replay_record}")
            success = bool(evaluator.n_success_trials > 0)
            logger.info(
                f"instance={idx} done: steps={steps} success={success} "
                f"trials={evaluator.n_trials} success_trials={evaluator.n_success_trials}"
            )
            metrics_out = {}
            for metric in evaluator.metrics:
                metrics_out.update(metric.gather_results())
            result = {
                "task": args.task_name,
                "task_id": TASK_NAMES_TO_INDICES[args.task_name],
                "instance_id": idx,
                "steps": steps,
                "success": success,
                "video": video_name,
                **metrics_out,
            }
            with open(metrics_path / f"{args.task_name}_{idx}_0.json", "w") as f:
                json.dump(result, f, indent=2, default=float)
            results.append(result)
            if config.write_video:
                evaluator.video_writer = None
                logger.info(f"Saved video to {video_name}")

    logger.info("=" * 60)
    for r in results:
        q = r.get("q_score", {}).get("final") if isinstance(r.get("q_score"), dict) else r.get("q_score")
        logger.info(f"RESULT task={r['task']} instance={r['instance_id']} success={r['success']} q_score={q}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
