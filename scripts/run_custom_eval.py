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
    # 2026-10-10: covered-57-60-69-93 joint ckpt (B100_task_misc.csv)
    "sweeping_garage": 57,
    "scrubbing_bathroom_floor": 60,
    "vacuuming_floors": 69,
    "clean_a_keyboard": 93,
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

# 2026-10-10: the bddl3 bundled with BEHAVIOR-1K-v3.7 has older definitions of these covered-57-60-69-93 tasks whose
# objects don't match the 2026 instance caches (vacuuming_floors: extra ashcan -> "ashcan.n.01_1 should exist in cached
# metadata" crash; sweeping_garage: extra pallet; clean_a_keyboard: rag/sink/soap version vs desk/dust/pipe_cleaner).
# The newer /workspace/persistent/BEHAVIOR-1K/bddl3 copies match the caches exactly -> use those.
_BDDL_2026_DIR = "/workspace/persistent/BEHAVIOR-1K/bddl3/bddl/activity_definitions"
for _t in ("sweeping_garage", "vacuuming_floors", "clean_a_keyboard"):
    _f = os.path.join(_BDDL_2026_DIR, _t, "problem0.bddl")
    if _t not in TASK_BDDL_OVERRIDES and os.path.isfile(_f):
        with open(_f) as _fh:
            # newer files declare (:domain behavior-1k); the v3.7 parser only accepts omnigibson
            TASK_BDDL_OVERRIDES[_t] = _fh.read().replace("(:domain behavior-1k)", "(:domain omnigibson)")


def _load_available_tasks_2026():
    with open(os.path.join(DATASET_2026_ROOT, "metadata", "available_tasks.yaml")) as f:
        return yaml.safe_load(f)


def _get_task_instance_path_2026(scene_name):
    # B1K_SPLIT=public_test reads the official public test instances (scene_test/public, ids 301-320;
    # the challenge reports public indices 0-9 = ids 301-310). Default: the training instances (scenes/).
    split = os.environ.get("B1K_SPLIT", "train")
    sub = {"train": "scenes", "public_test": os.path.join("scene_test", "public")}[split]
    scenes_path = os.path.join(DATASET_2026_ROOT, sub)
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

    # 2026-10-11 (user, task 35, SIM-ONLY ASSIST): B1K_ATTACH_ASSIST="child_inst>parent_inst" (e.g.
    # digital_camera.n.01_1>camera_tripod.n.01_1). When the held child's attachment (male) link is within
    # B1K_ASSIST_RADIUS of the parent's female link, the client overrides the L arm: resolved-rate IK on the L end effector
    # (robot control dict "eef_left_jacobian_relative", damped least squares) drives the child's attachment link to
    # B1K_ASSIST_HOVER m above the parent's and rotates it to the parent's orientation, base held still, L gripper closed.
    # Once within B1K_ASSIST_POS_OK m and B1K_ASSIST_ORN_OK deg for 5 steps, the L gripper opens for 40 steps (arm held)
    # so the child drops onto the mount -> AttachedTo checks 5 cm / 15 deg on that new contact. Then control returns to
    # the model. Uses privileged sim poses: a diagnostic/demonstration aid, not a submission feature. Logs "ASSIST:".
    _assist_spec = os.environ.get("B1K_ATTACH_ASSIST", "")
    if _assist_spec:
        from omnigibson.object_states import AttachedTo as _AT
        _aR = float(os.environ.get("B1K_ASSIST_RADIUS", "0.15"))
        _aH = float(os.environ.get("B1K_ASSIST_HOVER", "0.02"))
        _aPos = float(os.environ.get("B1K_ASSIST_POS_OK", "0.012"))
        _aOrn = float(os.environ.get("B1K_ASSIST_ORN_OK", "6.0"))
        _aMax = int(os.environ.get("B1K_ASSIST_MAX", "900"))
        # 2026-10-12 (user, task 63 "oracle run"): B1K_ASSIST_HOVER_DIR=robot - hover B1K_ASSIST_HOVER m out from the parent
        # link horizontally toward the robot (wall mount: detector -> nail) instead of above it; once aligned, "push": the
        # target moves 2 mm/step to B1K_ASSIST_PUSH m (0.01) past the parent link, orientation kept, until AttachedTo is true
        # (then release) or 60 steps (then release anyway).
        _aDir = os.environ.get("B1K_ASSIST_HOVER_DIR", "up")
        _aPush = float(os.environ.get("B1K_ASSIST_PUSH", "0.01"))
        # 05:45 (ORACLE run 1: orientation 0.0 deg within 50 steps but the hand stalled 8.5-9 cm short - base 1.1 m from the
        # nail, arm maxed at ~0.87 m): B1K_ASSIST_REACH=m - while aligning/pushing, if the target is more than m (horizontal)
        # from the base centre, the base creeps toward it (<= 0.15, robot frame), like the door grab's creep. 0 = off.
        _aReach = float(os.environ.get("B1K_ASSIST_REACH", "0"))
        # 06:35 (ORACLE2: 0.5 cm / 0.1 deg reached and pushed, attached=False). AttachedTo only tests on a NEW contact
        # (CONTACT_FOUND); if the detector already touched the nail while misaligned during the approach, the contact never
        # restarts. B1K_ASSIST_RETRIES=n: after a push without attachment, back out to 5 cm, then push again (n tries);
        # log the detector's contact bodies at each push end.
        _aRetries = int(os.environ.get("B1K_ASSIST_RETRIES", "0"))
        try:
            from omnigibson.object_states import ContactBodies as _CB
        except Exception:
            _CB = None
        _aChild, _aParent = _assist_spec.split(">")
        _orig_eval_step = eval_mod.Evaluator.step

        def _rotvec_from_mats(Rm):
            # axis * angle for rotation matrix Rm (world frame)
            ang = th.arccos(th.clamp((th.trace(Rm) - 1) / 2, -1.0, 1.0))
            if float(ang) < 1e-6:
                return th.zeros(3)
            ax = th.stack([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]]) / (2 * th.sin(ang))
            return ax * ang

        def _tt(x):
            # sim getters may hand back numpy or torch depending on the backend -> always float32 torch
            return th.as_tensor(x.cpu().numpy() if hasattr(x, "cpu") else x, dtype=th.float32)

        def _assist_step(self):
            st = getattr(self, "_as", None)
            if st is None:
                st = self._as = {"mode": "idle", "n": 0, "ok": 0, "rel": 0, "q": None, "done": False, "links": None}
            scope = self.env.task.object_scope or {}
            child, parent = scope.get(_aChild), scope.get(_aParent)
            if st["done"] or child is None or parent is None:
                return _orig_eval_step(self)
            if st["links"] is None:
                cand = child.states[_AT]._get_parent_candidates(parent)
                cl, pls = next(iter(cand.items()))
                st["links"] = (child.states[_AT].links[cl], parent.states[_AT].links[next(iter(pls))])
            clink, plink = st["links"]
            cp, co = (_tt(v) for v in clink.get_position_orientation())
            pp, po = (_tt(v) for v in plink.get_position_orientation())
            if _aDir == "robot":
                d = _tt(self.robot.get_position_orientation()[0]) - pp
                d[2] = 0.0
                d = d / th.linalg.norm(d).clamp(min=1e-6)
                if st["mode"] == "push":
                    st["push"] = max(-_aPush, st.get("push", _aH) - 0.002)
                    target = pp + d * st["push"]
                elif st["mode"] == "retreat":
                    target = pp + d * 0.05
                else:
                    target = pp + d * _aH
            else:
                target = pp + th.tensor([0.0, 0.0, _aH])
            e_p = target - cp
            Rc, Rp = _tt(T.quat2mat(co)), _tt(T.quat2mat(po))
            w = _rotvec_from_mats(Rp @ Rc.T)
            orn_deg = float(th.linalg.norm(w)) * 57.29578
            dist = float(th.linalg.norm(pp - cp))
            robot = self.robot
            jq = _tt(robot.get_joint_positions())
            l_idx = th.as_tensor(_tt(robot.arm_control_idx["left"]), dtype=th.long)
            # proprio gripper width is what the policy sees; here use the finger joints: holding = not fully open/closed
            if st["mode"] == "idle":
                if dist < _aR and not bool(child.states[_AT].get_value(parent)):
                    st["mode"] = "align"; st["n"] = 0
                    print(f"ASSIST: start align at step {self.env._current_step} (gap {dist:.3f} m, {orn_deg:.1f} deg)", flush=True)
                else:
                    return _orig_eval_step(self)
            # get the model's action, then override base + L arm + L gripper
            self.robot_action = self.policy.forward(obs=self.obs)
            a = self.robot_action
            a[0:3] = 0.0
            if st["mode"] == "retreat":
                st["rn"] = st.get("rn", 0) + 1
                if st["rn"] >= 30:
                    st["mode"] = "push"; st["push"] = 0.05; st["pn"] = 0
            if st["mode"] in ("align", "push", "retreat"):
                st["n"] += 1
                bpos_a, bq = (_tt(v) for v in robot.get_position_orientation())
                Rb = _tt(T.quat2mat(bq))
                eef_pos = _tt(robot.get_eef_position(arm="left"))
                if _aReach > 0:
                    hz = target[:2] - bpos_a[:2]
                    hd = float(th.linalg.norm(hz))
                    if hd > _aReach and float(th.linalg.norm(e_p)) > 0.01:
                        yaw = float(T.quat2euler(bq)[2])
                        ang_b = float(th.atan2(hz[1], hz[0])) - yaw
                        sp = min(0.15, 0.5 * (hd - _aReach))
                        a[0] = sp * float(th.cos(th.tensor(ang_b))); a[1] = sp * float(th.sin(th.tensor(ang_b)))
                        st["creep"] = st.get("creep", 0) + 1
                        if st["creep"] in (1, 50) or st["creep"] % 200 == 0:
                            print(f"ASSIST: base creep #{st['creep']} at step {self.env._current_step} (target {hd:.2f} m "
                                  f"from the base > {_aReach:.2f})", flush=True)
                v_lin = th.clamp(1.0 * e_p, -0.01, 0.01)
                om = w * min(1.0, (3.0 / 57.29578) / max(float(th.linalg.norm(w)), 1e-6))
                v_eef = v_lin - th.linalg.cross(om, cp - eef_pos)
                twist = th.cat([Rb.T @ v_eef, Rb.T @ om])
                J = _tt(robot.get_control_dict()["eef_left_jacobian_relative"])[:, l_idx]
                lam = 0.05
                dq = J.T @ th.linalg.solve(J @ J.T + lam * lam * th.eye(6), twist)
                q_cmd = jq[l_idx] + dq
                if st["n"] <= 3 or st["n"] % 100 == 0:
                    print(f"ASSIST-DBG: n={st['n']} l_idx={l_idx.tolist()} jq_arm={[round(float(v), 3) for v in jq[l_idx]]} "
                          f"act_arm_model={[round(float(v), 3) for v in a[7:14]]} J_shape={tuple(J.shape)} "
                          f"J_colnorm={[round(float(v), 3) for v in th.linalg.norm(J, dim=0)]} twist={[round(float(v), 4) for v in twist]} "
                          f"dq={[round(float(v), 4) for v in dq]} full_J={tuple(robot.get_control_dict()['eef_left_jacobian_relative'].shape)} "
                          f"n_joints={robot.n_joints}", flush=True)
                st["q"] = q_cmd
                a[7:14] = q_cmd.to(a.dtype) if hasattr(a, "dtype") and hasattr(a, "cpu") else q_cmd.numpy()
                a[14] = -1.0
                if float(th.linalg.norm(e_p)) < _aPos and orn_deg < _aOrn:
                    st["ok"] += 1
                else:
                    st["ok"] = 0
                if st["n"] % 50 == 1:
                    print(f"ASSIST: align step {st['n']} pos err {float(th.linalg.norm(e_p)):.3f} m, orn {orn_deg:.1f} deg", flush=True)
                if st["mode"] == "push":
                    st["pn"] = st.get("pn", 0) + 1
                    att_now = bool(child.states[_AT].get_value(parent))
                    if att_now or st["pn"] >= 60:
                        cb = ""
                        if _CB is not None and _CB in child.states:
                            try:
                                cb = sorted({getattr(b, "name", str(b)) for b in child.states[_CB].get_value()})
                            except Exception as e:
                                cb = f"? {e}"
                        tries = st.get("tries", 0) + 1; st["tries"] = tries
                        if not att_now and tries <= _aRetries:
                            print(f"ASSIST: push {tries} done at step {self.env._current_step} (attached=False, pos "
                                  f"{dist:.3f} m, orn {orn_deg:.1f}; touching {cb}) -> backing out to re-contact", flush=True)
                            st["mode"] = "retreat"; st["rn"] = 0
                        else:
                            print(f"ASSIST: push done at step {self.env._current_step} (attached={att_now}, pos "
                                  f"{dist:.3f} m to the parent link, orn {orn_deg:.1f}; touching {cb}) -> opening L gripper",
                                  flush=True)
                            st["mode"] = "release"; st["rel"] = 0
                elif st["mode"] == "retreat":
                    pass
                elif st["ok"] >= 5 and _aDir == "robot":
                    st["mode"] = "push"; st["push"] = _aH; st["pn"] = 0
                    print(f"ASSIST: aligned at step {self.env._current_step} (pos {float(th.linalg.norm(e_p)):.3f}, "
                          f"orn {orn_deg:.1f}) -> pushing onto the parent", flush=True)
                elif st["ok"] >= 5:
                    st["mode"] = "release"; st["rel"] = 0
                    print(f"ASSIST: aligned at step {self.env._current_step} (pos {float(th.linalg.norm(e_p)):.3f}, "
                                f"orn {orn_deg:.1f}) -> opening L gripper", flush=True)
                elif st["n"] >= _aMax:
                    print(f"ASSIST: gave up after {_aMax} steps (pos {float(th.linalg.norm(e_p)):.3f}, orn {orn_deg:.1f})", flush=True)
                    st["done"] = True; st["mode"] = "idle"
            if st["mode"] == "release":
                st["rel"] += 1
                a[7:14] = st["q"].to(a.dtype) if hasattr(a, "cpu") else st["q"].numpy()
                a[14] = 1.0
                if st["rel"] >= 40:
                    att = bool(child.states[_AT].get_value(parent))
                    print(f"ASSIST: released; attached={att} at step {self.env._current_step}", flush=True)
                    st["done"] = True
            obs, _, terminated, truncated, info = self.env.step(self.robot_action, n_render_iterations=1)
            self.obs = self._preprocess_obs(obs)
            if terminated or truncated:
                self.n_trials += 1
                if info["done"]["success"]:
                    self.n_success_trials += 1
            for metric in self.metrics:
                metric.step_callback(self.env)
            return terminated, truncated

        eval_mod.Evaluator.step = _assist_step

    # 2026-10-11 (user, task 78, SIM-ONLY DOOR CONTROLLER): B1K_DOOR_APPROACH="cabinet_door.n.01_1". Uses privileged sim
    # poses (a diagnostic aid, not a submission feature). Phases, logged as "DOOR:":
    #   scan     - arms held tucked (R1Pro tucked pose = all arm joints 0), torso held at its start pose, grippers open,
    #              base turns in place (no translation) until it faces the door (|heading error| < 6 deg).
    #   approach - drives toward the door, still tucked, and stops B1K_DOOR_STANDOFF m (default 0.70) from the door's
    #              bbox centre, facing it. Max B1K_DOOR_APPROACH_MAX steps (1500), then hands over anyway.
    #   model    - the policy drives, with two guards while a hand is not grasping the door:
    #              slow reach: within B1K_DOOR_SLOW_R m (0.35) of the door, that arm's joint command moves only
    #                B1K_DOOR_SLOW (0.4) of the way from the measured joints to the model's target each step;
    #              centre gate: within B1K_DOOR_GATE_R m (0.12) of the door, the nearest point of the door's box must lie
    #                within B1K_DOOR_MARGIN m (0.04) of the gripper camera axis (end-effector z = finger direction).
    #                Otherwise that gripper is held open and the arm is moved sideways only (damped least-squares IK,
    #                <= 1 cm/step, no forward motion) to centre the door, for at most 200 steps (then a 300-step
    #                cool-down with the model in charge).
    _door_inst = os.environ.get("B1K_DOOR_APPROACH", "")
    if _door_inst:
        import math as _m
        _dStand = float(os.environ.get("B1K_DOOR_STANDOFF", "0.70"))
        _dAppMax = int(os.environ.get("B1K_DOOR_APPROACH_MAX", "1500"))
        _dSlowR = float(os.environ.get("B1K_DOOR_SLOW_R", "0.35"))
        _dSlow = float(os.environ.get("B1K_DOOR_SLOW", "0.4"))
        _dGateR = float(os.environ.get("B1K_DOOR_GATE_R", "0.12"))
        _dMargin = float(os.environ.get("B1K_DOOR_MARGIN", "0.04"))
        # Scripted grab (2026-10-11, user): B1K_DOOR_GRAB=1 (default). When a hand that is not holding the door comes
        # within B1K_DOOR_GATE_R of it (and is not cooling down), the controller takes that arm (base stopped, torso held):
        #   align   - slide sideways to put the door on the camera axis (finger midpoint) and turn the wrist about that axis
        #             so the fingers close across the door's thinnest dimension (<= 1 cm and 3 deg per step, max 80 steps)
        #   insert  - move straight in along the finger direction, 0.5 cm/step, until the finger midpoint is 1.5 cm past the
        #             door surface (max 60 steps)
        #   close   - gripper closed, arm held, 30 steps; then is_grasping(door)?
        #   hold    - yes: gripper kept closed for 150 more steps while the model moves, then the model is in charge
        #   retreat - no: open, back out along the finger axis 15 steps, then a 300-step cool-down on that arm.
        # At most B1K_DOOR_GRAB_TRIES (4) grabs per episode. Logs "DOOR-GRAB:".
        _dGrab = os.environ.get("B1K_DOOR_GRAB", "1") == "1"
        _dTries = int(os.environ.get("B1K_DOOR_GRAB_TRIES", "4"))
        # v2 (2026-10-11, after the 17500 run): base creeps toward the grasp point while it is > B1K_DOOR_REACH m (0.60)
        # from the base centre; the grasp point is the top-edge centre; joint-limit clipping; STALL log lines.
        _dReach = float(os.environ.get("B1K_DOOR_REACH", "0.60"))
        # v3 (19:10, after DOORGRAB2: the model drove the base 0.45 m into the door right after the hand-over and knocked it
        # flat at ~step 450, so no hand ever came within B1K_DOOR_GATE_R and v2 never ran):
        #   B1K_DOOR_GRAB_EARLY=1 - start the scripted grab as soon as the base is at the standoff (and again 30 steps after
        #                           each failed try) instead of waiting for the model's hand; only while the door is upright.
        #   B1K_DOOR_KEEPOUT=m    - in model mode (no grab running) the base may not move closer than m (0.55) to the door
        #                           centre: the toward-door part of the model's base velocity is removed.
        _dEarly = os.environ.get("B1K_DOOR_GRAB_EARLY", "0") == "1"
        _dKeep = float(os.environ.get("B1K_DOOR_KEEPOUT", "0"))
        _dSide = os.environ.get("B1K_DOOR_SIDESTEP", "0") == "1"
        # 2026-10-12 (user, t78): "turn off the spinning rule ... the model base knows where to go". B1K_DOOR_SCAN=0 skips the
        # scan (spin in place until facing the door) and the tucked approach: the episode starts in "model" mode, so only the
        # model-mode guards (keep-out, slow reach, centre gate, grab - whichever are on) apply from step 0.
        _dScan = os.environ.get("B1K_DOOR_SCAN", "1") == "1"
        _ARM = {"left": (slice(7, 14), 14), "right": (slice(15, 22), 22)}
        _orig_eval_step_d = eval_mod.Evaluator.step

        def _dt(x):
            return th.as_tensor(x.cpu().numpy() if hasattr(x, "cpu") else x, dtype=th.float32)

        def _dset(a, sl, v):
            v = _dt(v)
            a[sl] = v.to(a.dtype) if hasattr(a, "cpu") else v.numpy()

        def _d_ik(robot, arm, idx, prev, bq, v_lin, om):
            Rb = _dt(T.quat2mat(bq))
            twist = th.cat([Rb.T @ v_lin, Rb.T @ om])
            J = _dt(robot.get_control_dict()[f"eef_{arm}_jacobian_relative"])[:, idx]
            q = prev + J.T @ th.linalg.solve(J @ J.T + 0.0025 * th.eye(6), twist)
            # clip to the joint limits so the integrated command cannot wind up past a limit (17500 run: hand frozen
            # for the whole 60-step insert while the command kept integrating)
            try:
                lo, hi = _dt(robot.joint_lower_limits)[idx], _dt(robot.joint_upper_limits)[idx]
                q = th.minimum(th.maximum(q, lo + 0.01), hi - 0.01)
            except Exception:
                pass
            return q

        def _d_top(c, Rd, ext):
            # grasp point = centre of the door's top edge, 2 cm below it (door stands upright: 2.4 x 26 x 46 cm)
            k = int(th.argmax(Rd[2, :].abs()))
            up = Rd[:, k] * (1.0 if float(Rd[2, k]) > 0 else -1.0)
            return c + up * (float(ext[k]) / 2 - 0.02)

        def _d_near(c, Rd, ext, p):
            loc = Rd.T @ (p - c)
            inside = bool(th.all(loc.abs() <= ext / 2))
            near = c + Rd @ th.maximum(th.minimum(loc, ext / 2), -ext / 2)
            depth = float(th.min(ext / 2 - loc.abs())) if inside else 0.0
            return near, inside, depth

        def _grab_step(self, st, door, a, jq, bq, c, Rd, ext, t):
            robot = self.robot
            G = st["grab"]; arm = G["arm"]; sl, g = _ARM[arm]
            idx = _dt(robot.arm_control_idx[arm]).long()
            G["n"] += 1
            if G["phase"] == "hold":                    # model drives base / torso / arm, gripper kept closed
                a[g] = -1.0
                if G["n"] >= 150:
                    print(f"DOOR-GRAB: {arm} hand-over to the model at step {t}", flush=True)
                    st["grab"] = None
                    st["held_once"] = True       # v3: no more early grabs once the door has been held
                st["cmd"][arm] = None
                return
            a[0] = 0.0; a[1] = 0.0; a[2] = 0.0
            _dset(a, slice(3, 7), G["torso"])
            Re = _dt(T.quat2mat(_dt(robot.get_eef_orientation(arm=arm))))
            u = Re[:, 2]
            fl = robot.finger_links[arm]
            pm = th.stack([_dt(l.get_position_orientation()[0]) for l in fl]).mean(0)
            near, inside, depth = _d_near(c, Rd, ext, pm)
            # v2 (after the 17500 run): aim at the centre of the top edge, not the nearest box point (which was the
            # board's end corner when the hand came in 14 cm along its width)
            P = _d_top(c, Rd, ext)
            vec = P - pm
            lat = vec - (vec @ u) * u
            off = float(th.linalg.norm(lat))
            ph = G["phase"]
            # base creep: the target is out of comfortable reach with the base and torso frozen -> roll toward it
            bpos = _dt(self.robot.get_position_orientation()[0])
            hz = P[:2] - bpos[:2]
            hd = float(th.linalg.norm(hz))
            if ph in ("align", "insert") and hd > _dReach:
                yaw = float(T.quat2euler(bq)[2])
                ang_b = _m.atan2(float(hz[1]), float(hz[0])) - yaw
                s = min(0.15, 0.5 * (hd - _dReach))
                a[0] = s * _m.cos(ang_b); a[1] = s * _m.sin(ang_b)
                if G.get("creep", 0) == 0:
                    print(f"DOOR-GRAB: {arm} target {hd:.2f} m from the base > {_dReach:.2f} -> base creep", flush=True)
                G["creep"] = G.get("creep", 0) + 1
            # v5 (20:55, after DOORGRAB4 on 22500: every try stalled ~15 cm SIDEWAYS of the target with arm joint 1 at its limit,
            # whatever the base distance -> the arm cannot bring the hand in toward the base centreline). B1K_DOOR_SIDESTEP=1:
            # during align/insert the base also slides sideways (robot-frame y) to cancel the horizontal sideways offset,
            # <= 0.1 (0.075 m/s), so the door ends up in front of the grasping arm's shoulder.
            if _dSide and ph in ("align", "insert") and off > 0.02:
                lat_b = _dt(T.quat2mat(bq)).T @ lat
                a[1] = float(a[1]) + max(-0.1, min(0.1, 1.5 * float(lat_b[1])))
                if G.get("side", 0) == 0:
                    print(f"DOOR-GRAB: {arm} sidestep on ({float(lat_b[1]) * 100:+.1f} cm sideways in the base frame)", flush=True)
                G["side"] = G.get("side", 0) + 1
            # stall detection: the finger midpoint has barely moved over the last 15 steps of align/insert
            if ph in ("align", "insert"):
                if G["n"] % 15 == 0:
                    if G.get("pm15") is not None and float(th.linalg.norm(pm - G["pm15"])) < 0.003 and off > 0.01:
                        G["stall"] = G.get("stall", 0) + 1
                        try:
                            lo, hi = _dt(robot.joint_lower_limits)[idx], _dt(robot.joint_upper_limits)[idx]
                            jm = _dt(robot.get_joint_positions())[idx]
                            marg = th.minimum(jm - lo, hi - jm)
                            lim = f"min joint-limit margin {float(marg.min()):.3f} rad (joint {int(th.argmin(marg))})"
                        except Exception:
                            lim = "limits n/a"
                        print(f"DOOR-GRAB: {arm} STALL in {ph} at step {t} ({off * 100:.1f} cm off, "
                              f"{float(th.linalg.norm(vec)) * 100:.1f} cm to target, {lim})", flush=True)
                    G["pm15"] = pm.clone()
            if ph == "align":
                nrm = Rd[:, int(th.argmin(ext))]
                n_p = nrm - (nrm @ u) * u
                if float(th.linalg.norm(n_p)) > 1e-3:
                    n_p = n_p / th.linalg.norm(n_p)
                    y = Re[:, 1]
                    ang = _m.atan2(float(u @ th.linalg.cross(y, n_p)), float(y @ n_p))
                    if ang > _m.pi / 2:
                        ang -= _m.pi
                    elif ang < -_m.pi / 2:
                        ang += _m.pi
                else:
                    ang = 0.0
                om = u * max(-_m.radians(3), min(_m.radians(3), ang))
                # v2: also tilt the fingers to point straight down (top grasp on the upright board), <= 3 deg/step
                dn = th.tensor([0.0, 0.0, -1.0])
                tax = th.linalg.cross(u, dn)
                tilt = _m.atan2(float(th.linalg.norm(tax)), float(u @ dn))
                if float(th.linalg.norm(tax)) > 1e-6:
                    om = om + tax / th.linalg.norm(tax) * min(_m.radians(3), tilt)
                v = lat * min(1.0, 0.01 / max(off, 1e-6))
                G["q"] = _d_ik(robot, arm, idx, G["q"], bq, v, om)
                a[g] = 1.0
                if G["n"] == 1:
                    print(f"DOOR-GRAB: {arm} align start: {off * 100:.1f} cm off the top-edge centre, wrist "
                          f"{_m.degrees(ang):+.0f} deg, fingers {_m.degrees(tilt):.0f} deg from straight down", flush=True)
                if off < 0.01 and abs(ang) < _m.radians(10) and tilt < _m.radians(15):
                    G["phase"] = "insert"; G["n"] = 0
                    print(f"DOOR-GRAB: {arm} aligned at step {t} ({off * 100:.1f} cm off, wrist {_m.degrees(ang):+.0f} deg) -> insert", flush=True)
                elif G["n"] >= 120:
                    G["phase"] = "insert"; G["n"] = 0
                    print(f"DOOR-GRAB: {arm} align timeout at step {t} ({off * 100:.1f} cm off, wrist {_m.degrees(ang):+.0f} deg) -> insert anyway", flush=True)
            elif ph == "insert":
                # v2: straight at the grasp point (2 cm below the top-edge centre), 0.5 cm/step. The old "depth >= 1.5 cm"
                # test could never pass: the board is only 2.4 cm thick, so the depth inside it is at most 1.2 cm.
                dv = float(th.linalg.norm(vec))
                v = vec * min(1.0, 0.005 / max(dv, 1e-6))
                G["q"] = _d_ik(robot, arm, idx, G["q"], bq, v, th.zeros(3))
                a[g] = 1.0
                if dv < 0.01 or G["n"] >= 80:
                    print(f"DOOR-GRAB: {arm} inserted at step {t} ({dv * 100:.1f} cm from the grasp point, inside box {inside}, "
                          f"{G['n']} steps) -> close", flush=True)
                    G["phase"] = "close"; G["n"] = 0
            elif ph == "close":
                a[g] = -1.0
                if G["n"] >= 30:
                    try:
                        ok = str(robot.is_grasping(arm=arm, candidate_obj=door)).endswith("TRUE")
                    except Exception:
                        ok = False
                    print(f"DOOR-GRAB: {arm} closed at step {t}: grasping door = {ok}", flush=True)
                    G["phase"] = "hold" if ok else "retreat"; G["n"] = 0
            elif ph == "retreat":
                a[g] = 1.0
                if G["n"] > 15:
                    G["q"] = _d_ik(robot, arm, idx, G["q"], bq, -u * 0.005, th.zeros(3))
                if G["n"] >= 30:
                    print(f"DOOR-GRAB: {arm} retreated at step {t} -> 300-step cool-down", flush=True)
                    st["cool"][arm] = 300; st["grab"] = None; st["cmd"][arm] = None
                    return
            _dset(a, sl, G["q"])

        def _door_step(self):
            scope = self.env.task.object_scope or {}
            door = scope.get(_door_inst)
            if door is None:
                return _orig_eval_step_d(self)
            robot = self.robot
            t = int(self.env._current_step)
            st = getattr(self, "_dr", None)
            if st is None or t < st["t"]:
                jq = _dt(robot.get_joint_positions())
                st = self._dr = {"mode": "scan" if _dScan else "model", "t": t, "n": 0, "torso": jq[_dt(robot.trunk_control_idx).long()].clone(),
                                 "bbox": None, "bbox_t": -99, "gate": {"left": 0, "right": 0},
                                 "cool": {"left": 0, "right": 0}, "slow_n": 0, "cmd": {"left": None, "right": None},
                                 "grab": None, "tries": 0}
                print(f"DOOR: episode start (step {t}) -> {st['mode']}", flush=True)
            st["t"] = t
            if st["bbox"] is None or t - st["bbox_t"] >= 10:
                c, q, e, _ = door.get_base_aligned_bbox()
                st["bbox"] = (_dt(c), _dt(T.quat2mat(_dt(q))), _dt(e)); st["bbox_t"] = t
            c, Rd, ext = st["bbox"]
            bpos, bq = (_dt(v) for v in robot.get_position_orientation())
            yaw = float(T.quat2euler(bq)[2])
            dx, dy = float(c[0] - bpos[0]), float(c[1] - bpos[1])
            dist = _m.hypot(dx, dy)
            err = (_m.atan2(dy, dx) - yaw + _m.pi) % (2 * _m.pi) - _m.pi
            self.robot_action = self.policy.forward(obs=self.obs)
            a = self.robot_action
            jq = _dt(robot.get_joint_positions())
            st["n"] += 1
            if st["mode"] in ("scan", "approach"):
                for arm, (sl, g) in _ARM.items():
                    _dset(a, sl, th.zeros(7)); a[g] = 1.0
                _dset(a, slice(3, 7), st["torso"])
                w = max(-0.5, min(0.5, 1.5 * err))
                if st["mode"] == "scan":
                    if abs(w) < 0.15:
                        w = 0.15 if w >= 0 else -0.15
                    a[0] = 0.0; a[1] = 0.0; a[2] = w
                    if abs(err) < _m.radians(6):
                        st["mode"] = "approach"; st["n"] = 0
                        print(f"DOOR: facing the door at step {t} (dist {dist:.2f} m) -> approach", flush=True)
                else:
                    v = max(-0.3, min(0.4, 0.8 * (dist - _dStand)))
                    a[0] = v * _m.cos(err); a[1] = v * _m.sin(err); a[2] = w
                    if (abs(dist - _dStand) < 0.03 and abs(err) < _m.radians(8)) or st["n"] >= _dAppMax:
                        print(f"DOOR: at standoff at step {t} (dist {dist:.2f} m, heading err {_m.degrees(err):.1f} deg"
                              f"{', TIMEOUT' if st['n'] >= _dAppMax else ''}) -> model", flush=True)
                        st["mode"] = "model"; st["n"] = 0
            elif st["grab"] is not None:
                _grab_step(self, st, door, a, jq, bq, c, Rd, ext, t)
                if st["grab"] is None:
                    st["early_wait"] = 30
            elif (_dEarly and _dGrab and st["tries"] < _dTries and st.get("early_wait", 0) <= 0 and not st.get("held_once")
                  and abs(float(Rd[2, int(th.argmax(ext))])) > 0.8):
                P = _d_top(c, Rd, ext)
                arm = min(_ARM, key=lambda k: float(th.linalg.norm(_dt(robot.get_eef_position(arm=k)) - P)))
                sl, g = _ARM[arm]
                idx = _dt(robot.arm_control_idx[arm]).long()
                prev = st["cmd"][arm] if st["cmd"][arm] is not None else jq[idx].clone()
                st["tries"] += 1
                st["grab"] = {"arm": arm, "phase": "align", "n": 0, "q": prev.clone(),
                              "torso": jq[_dt(robot.trunk_control_idx).long()].clone()}
                print(f"DOOR-GRAB: early try {st['tries']} with {arm} at step {t} (door upright, "
                      f"{float(th.linalg.norm(_dt(robot.get_eef_position(arm=arm)) - P)):.2f} m from the grasp point) -> align",
                      flush=True)
                for k, (sl2, g2) in _ARM.items():
                    q2 = st["cmd"][k] if st["cmd"][k] is not None else jq[_dt(robot.arm_control_idx[k]).long()]
                    _dset(a, sl2, q2); a[g2] = 1.0
                a[0] = 0.0; a[1] = 0.0; a[2] = 0.0
                _dset(a, slice(3, 7), st["grab"]["torso"])
            else:
                for arm, (sl, g) in _ARM.items():
                    if st["grab"] is not None:
                        break
                    try:
                        held = str(robot.is_grasping(arm=arm, candidate_obj=door)).endswith("TRUE")
                    except Exception:
                        held = False
                    if held:
                        st["cmd"][arm] = None
                        continue
                    idx = _dt(robot.arm_control_idx[arm]).long()
                    ep = _dt(robot.get_eef_position(arm=arm))
                    loc = Rd.T @ (ep - c)
                    near = c + Rd @ th.maximum(th.minimum(loc, ext / 2), -ext / 2)
                    vec = near - ep
                    d_near = float(th.linalg.norm(vec))
                    # arm commands are integrated from the previous COMMAND (not the measured joints), so a
                    # position-controlled arm under gravity does not sag / stall at the measured pose
                    prev = st["cmd"][arm] if st["cmd"][arm] is not None else jq[idx].clone()
                    used = False
                    if d_near < _dSlowR:
                        tgt = _dt(a[sl])
                        _dset(a, sl, prev + _dSlow * (tgt - prev)); used = True
                        st["slow_n"] += 1
                        if st["slow_n"] % 200 == 1:
                            print(f"DOOR: slow reach ({arm}, {d_near:.2f} m from door) at step {t}", flush=True)
                    if st["cool"][arm] > 0:
                        st["cool"][arm] -= 1
                        st["cmd"][arm] = _dt(a[sl]).clone() if used else None
                        continue
                    if _dGrab and d_near < _dGateR and st["tries"] < _dTries:
                        st["tries"] += 1
                        st["grab"] = {"arm": arm, "phase": "align", "n": 0, "q": prev.clone(),
                                      "torso": jq[_dt(robot.trunk_control_idx).long()].clone()}
                        print(f"DOOR-GRAB: try {st['tries']} with {arm} at step {t} ({d_near:.2f} m from the door) -> align",
                              flush=True)
                        _dset(a, sl, prev); a[g] = 1.0
                        a[0] = 0.0; a[1] = 0.0; a[2] = 0.0
                        continue
                    if d_near < _dGateR:
                        u = _dt(T.quat2mat(_dt(robot.get_eef_orientation(arm=arm))))[:, 2]
                        lat = vec - (vec @ u) * u
                        off = float(th.linalg.norm(lat))
                        if off > _dMargin:
                            st["gate"][arm] += 1
                            Rb = _dt(T.quat2mat(bq))
                            v_lin = lat * min(1.0, 0.01 / max(off, 1e-6))
                            twist = th.cat([Rb.T @ v_lin, th.zeros(3)])
                            J = _dt(robot.get_control_dict()[f"eef_{arm}_jacobian_relative"])[:, idx]
                            dq = J.T @ th.linalg.solve(J @ J.T + 0.0025 * th.eye(6), twist)
                            _dset(a, sl, prev + dq); used = True
                            a[g] = 1.0
                            if st["gate"][arm] in (1, 50, 100, 150):
                                print(f"DOOR: centre gate {arm} at step {t}: door {off * 100:.1f} cm off the camera axis "
                                      f"({d_near:.2f} m away) -> gripper held open, centring", flush=True)
                            if st["gate"][arm] >= 200:
                                print(f"DOOR: centre gate {arm} gave up at step {t} ({off * 100:.1f} cm off) -> 300-step "
                                      f"cool-down", flush=True)
                                st["gate"][arm] = 0; st["cool"][arm] = 300
                        else:
                            if st["gate"][arm] > 0:
                                print(f"DOOR: centred {arm} at step {t} ({off * 100:.1f} cm off) -> model", flush=True)
                            st["gate"][arm] = 0
                    st["cmd"][arm] = _dt(a[sl]).clone() if used else None
            if st.get("early_wait", 0) > 0:
                st["early_wait"] -= 1
            if _dKeep > 0 and st["mode"] == "model" and st["grab"] is None and dist < _dKeep:
                tw = float(a[0]) * _m.cos(err) + float(a[1]) * _m.sin(err)
                held_any = False
                if tw > 0:
                    for k in _ARM:
                        try:
                            held_any = held_any or str(robot.is_grasping(arm=k, candidate_obj=door)).endswith("TRUE")
                        except Exception:
                            pass
                if tw > 0 and not held_any:
                    a[0] = float(a[0]) - tw * _m.cos(err); a[1] = float(a[1]) - tw * _m.sin(err)
                    st["keep_n"] = st.get("keep_n", 0) + 1
                    if st["keep_n"] in (1, 10, 100) or st["keep_n"] % 500 == 0:
                        print(f"DOOR: keep-out at step {t} (base {dist:.2f} m from the door, toward-door cmd {tw:.2f} removed, "
                              f"#{st['keep_n']})", flush=True)
            obs, _, terminated, truncated, info = self.env.step(self.robot_action, n_render_iterations=1)
            self.obs = self._preprocess_obs(obs)
            if terminated or truncated:
                self.n_trials += 1
                if info["done"]["success"]:
                    self.n_success_trials += 1
            for metric in self.metrics:
                metric.step_callback(self.env)
            return terminated, truncated

        eval_mod.Evaluator.step = _door_step


    # 2026-10-11 (user, task 35): SEAT-AND-TWIST. B1K_SEAT_TWIST=1. Every task-35 near miss had the camera upright and within
    # 0.3-4.3 cm of the tripod mount but turned 15-120 deg about the vertical axis; the sim attaches only when a NEW contact
    # starts while the gap is < 5 cm and < 15 deg. Robot-state only (no object poses):
    #   trigger - both grippers holding (finger width strictly inside (-0.9, 0.9) of its range while commanded closed) for
    #             B1K_SEAT_HOLD (300) consecutive steps (camera in L, tripod in R)
    #   cycle   - for each yaw offset in B1K_SEAT_ANGLES (+30,-30,+60,-60,+90,-90,+120,-120,+150,-150,180; relative to the
    #             start orientation): lift the L hand 3 cm (1 mm/step), turn it about the vertical axis to the offset
    #             (<= 1 deg/step), lower it up to 4.5 cm (1 mm/step; contact or 1.5 cm below the sweep-start height
    #             stops it), dwell 20 steps. Gripper kept closed,
    #             base stopped, torso held; R arm and everything else keep the model's (and server rules') output.
    #   stop    - a lift that raises the hand < 1 cm in 30 steps = the camera is locked to the tripod -> open L for 60 steps,
    #             then the model is in charge. Also stops after the last angle (hand back to the model). Once per episode.
    # IK = damped least squares on the robot's own eef Jacobian, integrated from the previous command. Logs "SEAT-TWIST:".
    if os.environ.get("B1K_SEAT_TWIST", "0") == "1":
        import math as _sm
        _sHold = int(os.environ.get("B1K_SEAT_HOLD", "300"))
        _sAng = [float(x) for x in os.environ.get("B1K_SEAT_ANGLES", "30,-30,60,-60,90,-90,120,-120,150,-150,180").split(",")]
        _orig_eval_step_s = eval_mod.Evaluator.step

        def _st(x):
            return th.as_tensor(x.cpu().numpy() if hasattr(x, "cpu") else x, dtype=th.float32)

        def _sset(a, sl, v):
            v = _st(v)
            a[sl] = v.to(a.dtype) if hasattr(a, "cpu") else v.numpy()

        def _s_width(robot, arm):
            gi = _st(robot.gripper_control_idx[arm]).long()
            q = _st(robot.get_joint_positions())[gi]
            lo, hi = _st(robot.joint_lower_limits)[gi], _st(robot.joint_upper_limits)[gi]
            return float((2 * (q - lo) / (hi - lo).clamp(min=1e-6) - 1).mean())

        def _s_ik(robot, idx, prev, bq, v_lin, om):
            Rb = _st(T.quat2mat(bq))
            twist = th.cat([Rb.T @ v_lin, Rb.T @ om])
            J = _st(robot.get_control_dict()["eef_left_jacobian_relative"])[:, idx]
            q = prev + J.T @ th.linalg.solve(J @ J.T + 0.0025 * th.eye(6), twist)
            lo, hi = _st(robot.joint_lower_limits)[idx], _st(robot.joint_upper_limits)[idx]
            return th.minimum(th.maximum(q, lo + 0.01), hi - 0.01)

        def _s_yaw(robot):
            Re = _st(T.quat2mat(_st(robot.get_eef_orientation(arm="left"))))
            x = Re[:, 0] if abs(float(Re[2, 0])) < abs(float(Re[2, 1])) else Re[:, 1]   # the more horizontal eef axis
            return _sm.atan2(float(x[1]), float(x[0]))

        def _seat_step(self):
            robot = self.robot
            t = int(self.env._current_step)
            s = getattr(self, "_seat", None)
            if s is None or t < s["t"]:
                s = self._seat = {"t": t, "hl": 0, "hr": 0, "mode": "wait", "q": None, "k": 0, "n": 0, "y0": 0.0,
                                  "z0": 0.0, "torso": None, "phase": None}
                print(f"SEAT-TWIST: episode start (step {t})", flush=True)
            s["t"] = t
            self.robot_action = self.policy.forward(obs=self.obs)
            a = self.robot_action
            if s["mode"] == "wait":
                s["hl"] = s["hl"] + 1 if (-0.9 < _s_width(robot, "left") < 0.9 and float(a[14]) < 0) else 0
                s["hr"] = s["hr"] + 1 if (-0.9 < _s_width(robot, "right") < 0.9 and float(a[22]) < 0) else 0
                if s["hl"] >= _sHold and s["hr"] >= _sHold:
                    jq = _st(robot.get_joint_positions())
                    idx = _st(robot.arm_control_idx["left"]).long()
                    s.update(mode="run", q=jq[idx].clone(), torso=jq[_st(robot.trunk_control_idx).long()].clone(),
                             k=0, n=0, phase="lift", y0=_s_yaw(robot), z0=float(_st(robot.get_eef_position(arm="left"))[2]))
                    print(f"SEAT-TWIST: both hands holding {_sHold} steps at step {t} -> sweep start (yaw0 "
                          f"{_sm.degrees(s['y0']):+.0f} deg)", flush=True)
            if s["mode"] in ("run", "release"):
                idx = _st(robot.arm_control_idx["left"]).long()
                bq = _st(robot.get_position_orientation()[1])
                a[0] = 0.0; a[1] = 0.0; a[2] = 0.0
                _sset(a, slice(3, 7), s["torso"])
                s["n"] += 1
                ez = float(_st(robot.get_eef_position(arm="left"))[2])
                zero = th.zeros(3)
                if s["mode"] == "release":
                    a[14] = 1.0
                    if s["n"] >= 60:
                        print(f"SEAT-TWIST: released, model in charge at step {t}", flush=True)
                        s["mode"] = "done"
                else:
                    a[14] = -1.0
                    ph = s["phase"]
                    if ph == "lift":
                        if s["n"] == 1:
                            s["zl"] = ez
                        s["q"] = _s_ik(robot, idx, s["q"], bq, th.tensor([0.0, 0.0, 0.001]), zero)
                        if s["n"] >= 30:
                            rise = ez - s["zl"]
                            if s["k"] > 0 and rise < 0.01:
                                print(f"SEAT-TWIST: lift blocked ({rise * 100:.1f} cm) at step {t} after angle "
                                      f"{_sAng[s['k'] - 1]:+.0f} -> camera locked? releasing L", flush=True)
                                s["mode"] = "release"; s["n"] = 0
                            elif s["k"] >= len(_sAng):
                                print(f"SEAT-TWIST: all {len(_sAng)} angles tried, model in charge at step {t}", flush=True)
                                s["mode"] = "done"
                            else:
                                s["phase"] = "turn"; s["n"] = 0
                    elif ph == "turn":
                        tgt = s["y0"] + _sm.radians(_sAng[s["k"]])
                        err = (tgt - _s_yaw(robot) + _sm.pi) % (2 * _sm.pi) - _sm.pi
                        w = max(-_sm.radians(1), min(_sm.radians(1), err))
                        s["q"] = _s_ik(robot, idx, s["q"], bq, zero, th.tensor([0.0, 0.0, w]))
                        if abs(err) < _sm.radians(2) or s["n"] >= 250:
                            s["phase"] = "lower"; s["n"] = 0
                    elif ph == "lower":
                        s["q"] = _s_ik(robot, idx, s["q"], bq, th.tensor([0.0, 0.0, -0.001]), zero)
                        # stop 1.5 cm below the sweep-start height so repeated cycles do not keep pushing down
                        if s["n"] >= 45 or ez <= s["z0"] - 0.015:
                            s["phase"] = "dwell"; s["n"] = 0
                    elif ph == "dwell":
                        if s["n"] >= 20:
                            print(f"SEAT-TWIST: angle {_sAng[s['k']]:+.0f} deg seated at step {t} (eef z {ez:.3f}, "
                                  f"yaw {_sm.degrees(_s_yaw(robot)):+.0f})", flush=True)
                            s["k"] += 1; s["phase"] = "lift"; s["n"] = 0
                    _sset(a, slice(7, 14), s["q"])
            obs, _, terminated, truncated, info = self.env.step(self.robot_action, n_render_iterations=1)
            self.obs = self._preprocess_obs(obs)
            if terminated or truncated:
                self.n_trials += 1
                if info["done"]["success"]:
                    self.n_success_trials += 1
            for metric in self.metrics:
                metric.step_callback(self.env)
            return terminated, truncated

        eval_mod.Evaluator.step = _seat_step
        print("SEAT-TWIST: rule on (task 35 client)", flush=True)

    # 2026-10-11 (user, task 78): CLOSE-BELOW gate. B1K_CLOSE_BELOW=<m> (e.g. 0.46 = the cabinet door's height). A gripper
    # may only close once its end effector is lower than <m> above the robot base (robot-body frame, own kinematics only):
    # while that hand's eef is higher, a close command (< 0) is replaced by open (+1). A hand that is already holding
    # (finger width strictly inside (-0.9, 0.9) of its range) is never forced open, so a carried door is not dropped when
    # it is lifted. Wraps whatever Evaluator.step is installed (plain, door controller, ...). Logs "CLOSE-BELOW:".
    _cb = os.environ.get("B1K_CLOSE_BELOW", "")
    if _cb:
        _cbZ = float(_cb)
        _orig_eval_step_cb = eval_mod.Evaluator.step

        def _cb_t(x):
            return th.as_tensor(x.cpu().numpy() if hasattr(x, "cpu") else x, dtype=th.float32)

        def _cb_holding(robot, arm):
            gi = _cb_t(robot.gripper_control_idx[arm]).long()
            q = _cb_t(robot.get_joint_positions())[gi]
            lo, hi = _cb_t(robot.joint_lower_limits)[gi], _cb_t(robot.joint_upper_limits)[gi]
            w = float((2 * (q - lo) / (hi - lo).clamp(min=1e-6) - 1).mean())
            return -0.9 < w < 0.9

        def _cb_step(self):
            pf = self.policy.forward
            robot = self.robot
            st = self.__dict__.setdefault("_cbs", {"n": 0, "pass": {"left": 0, "right": 0}})

            def _gated(*args, **kw):
                a = pf(*args, **kw)
                try:
                    bpos, bq = robot.get_position_orientation()
                    Rb = _cb_t(T.quat2mat(_cb_t(bq)))
                    for arm, g in (("left", 14), ("right", 22)):
                        if float(a[g]) >= 0:
                            continue
                        ez = float((Rb.T @ (_cb_t(robot.get_eef_position(arm=arm)) - _cb_t(bpos)))[2])
                        if ez < _cbZ:
                            st["pass"][arm] += 1
                            if st["pass"][arm] in (1, 100) or st["pass"][arm] % 1000 == 0:
                                print(f"CLOSE-BELOW: {arm} close allowed at step {self.env._current_step} (eef {ez:.2f} m "
                                      f"above the base < {_cbZ:.2f})", flush=True)
                        elif not _cb_holding(robot, arm):
                            a[g] = 1.0
                            st["n"] += 1
                            if st["n"] in (1, 10, 100) or st["n"] % 500 == 0:
                                print(f"CLOSE-BELOW: {arm} close blocked at step {self.env._current_step} (eef {ez:.2f} m "
                                      f"above the base >= {_cbZ:.2f}; #{st['n']})", flush=True)
                except Exception as e:
                    print(f"CLOSE-BELOW: error {e}", flush=True)
                return a

            self.policy.forward = _gated
            try:
                return _orig_eval_step_cb(self)
            finally:
                self.policy.forward = pf

        eval_mod.Evaluator.step = _cb_step
        print(f"CLOSE-BELOW: rule on (close only when eef < {_cbZ:.2f} m above the base)", flush=True)

    # 2026-10-14 (user: "yes" to the task-0 press rule). PRESS-ASSIST, B1K_T0_PRESS=1. Public t0 pass (cont5000, 0.50):
    # 304 and 309 grasped and lifted the radio with R, but the pressing L hand stopped 13-16 cm from the radio centre with
    # the L hand >= 0.20 m from the R hand (successes press with CLOSED L fingers at L-R 0.13-0.19 m; 304's L was open 72%
    # of its press stage). While the policy's own stage (B1K_STAGE_FILE, written by serve_indep) is B1K_T0_PRESS_STAGE (3)
    # and R holds something: L gripper forced closed; after B1K_T0_PRESS_WAIT (60) steps in that stage, if the hands are
    # > B1K_T0_PRESS_D (0.15 m) apart, the L arm is driven by IK toward the R hand (orientation held) at B1K_T0_PRESS_V
    # (0.003 m/step) until <= D or stalled, held 15 steps (toggle needs 5 steps of finger overlap), backed off 4 cm, and
    # repeated up to B1K_T0_PRESS_POKES (6) times; then the model gets L back for 200 steps before pokes are allowed again.
    # Robot state + policy stage only. Logs "PRESS-ASSIST".
    if os.environ.get("B1K_T0_PRESS", "0") == "1":
        _paStage = int(os.environ.get("B1K_T0_PRESS_STAGE", "3")); _paWait = int(os.environ.get("B1K_T0_PRESS_WAIT", "60"))
        _paD = float(os.environ.get("B1K_T0_PRESS_D", "0.15")); _paV = float(os.environ.get("B1K_T0_PRESS_V", "0.003"))
        _paPokes = int(os.environ.get("B1K_T0_PRESS_POKES", "6")); _paFile = os.environ.get("B1K_STAGE_FILE", "")
        _orig_eval_step_pa = eval_mod.Evaluator.step

        def _pa_t(x):
            return th.as_tensor(x.cpu().numpy() if hasattr(x, "cpu") else x, dtype=th.float32)

        def _pa_width(robot, arm):
            gi = _pa_t(robot.gripper_control_idx[arm]).long()
            q = _pa_t(robot.get_joint_positions())[gi]
            lo, hi = _pa_t(robot.joint_lower_limits)[gi], _pa_t(robot.joint_upper_limits)[gi]
            return float((2 * (q - lo) / (hi - lo).clamp(min=1e-6) - 1).mean())

        def _pa_stage():
            try:
                with open(_paFile) as fh:
                    return int(fh.read().split()[1])
            except Exception:
                return -1

        def _pa_ik(robot, prev, Rb, v):
            idx = _pa_t(robot.arm_control_idx["left"]).long()
            twist = th.cat([Rb.T @ v, th.zeros(3)])
            J = _pa_t(robot.get_control_dict()["eef_left_jacobian_relative"])[:, idx]
            q = prev + J.T @ th.linalg.solve(J @ J.T + 0.0025 * th.eye(6), twist)
            lo, hi = _pa_t(robot.joint_lower_limits)[idx], _pa_t(robot.joint_upper_limits)[idx]
            return th.minimum(th.maximum(q, lo + 0.01), hi - 0.01)

        def _pa_step(self):
            pf = self.policy.forward
            robot = self.robot
            s = self.__dict__.setdefault("_pas", {"n3": 0, "ph": "model", "k": 0, "n": 0, "q": None, "cool": 0,
                                                  "closed": 0, "last": None, "t": -1})
            t = int(self.env._current_step)
            if t < s["t"]:
                s.update(n3=0, ph="model", k=0, n=0, q=None, cool=0, closed=0, last=None)
            s["t"] = t

            def _gated(*args, **kw):
                a = pf(*args, **kw)
                try:
                    stg = _pa_stage()
                    r_hold = -0.95 < _pa_width(robot, "right") < 0.9
                    if stg != _paStage or not r_hold:
                        if s["ph"] != "model":
                            print(f"PRESS-ASSIST: off at step {t} (stage {stg}, R holding {r_hold})", flush=True)
                        s.update(n3=0, ph="model", k=0, n=0, q=None)
                        return a
                    s["n3"] += 1
                    a[14] = -1.0
                    s["closed"] += 1
                    if s["closed"] in (1, 100) or s["closed"] % 1000 == 0:
                        print(f"PRESS-ASSIST: L kept closed in press stage {stg} at step {t} (#{s['closed']})", flush=True)
                    bpos, bq = robot.get_position_orientation()
                    Rb = _pa_t(T.quat2mat(_pa_t(bq)))
                    pL = _pa_t(robot.get_eef_position(arm="left")); pR = _pa_t(robot.get_eef_position(arm="right"))
                    d = float((pR - pL).norm()); u = (pR - pL) / max(d, 1e-6)
                    idx = _pa_t(robot.arm_control_idx["left"]).long()
                    if s["ph"] == "model":
                        if s["cool"] > 0:
                            s["cool"] -= 1
                        elif s["n3"] >= _paWait and d > _paD:
                            s.update(ph="in", n=0, q=_pa_t(robot.get_joint_positions())[idx].clone(), last=d)
                            s["k"] += 1
                            print(f"PRESS-ASSIST: poke #{s['k']} at step {t} (hands {d:.3f} m apart > {_paD:.2f}) -> "
                                  f"L toward R", flush=True)
                        if s["ph"] == "model":
                            return a
                    s["n"] += 1
                    if s["ph"] == "in":
                        s["q"] = _pa_ik(robot, s["q"], Rb, _paV * u)
                        stall = s["n"] % 15 == 0 and (s["last"] - d) < 0.002
                        if s["n"] % 15 == 0:
                            s["last"] = d
                        if d <= _paD or stall or s["n"] >= 120:
                            print(f"PRESS-ASSIST: poke #{s['k']} reached at step {t} (hands {d:.3f} m"
                                  f"{', stalled' if stall else ''}) -> hold", flush=True)
                            s.update(ph="hold", n=0)
                    elif s["ph"] == "hold":
                        if s["n"] >= 15:
                            s.update(ph="out", n=0)
                    elif s["ph"] == "out":
                        s["q"] = _pa_ik(robot, s["q"], Rb, -_paV * u)
                        if s["n"] >= 14:
                            if s["k"] >= _paPokes:
                                print(f"PRESS-ASSIST: {s['k']} pokes done at step {t} -> model for 200 steps", flush=True)
                                s.update(ph="model", k=0, n=0, cool=200)
                                return a
                            s.update(ph="in", n=0, last=d)
                            s["k"] += 1
                            print(f"PRESS-ASSIST: poke #{s['k']} at step {t} (hands {d:.3f} m)", flush=True)
                    v = s["q"].to(a.dtype) if hasattr(a, "cpu") else s["q"].numpy()
                    a[7:14] = v
                except Exception as e:
                    print(f"PRESS-ASSIST: error {e}", flush=True)
                return a

            self.policy.forward = _gated
            try:
                return _orig_eval_step_pa(self)
            finally:
                self.policy.forward = pf

        eval_mod.Evaluator.step = _pa_step
        print(f"PRESS-ASSIST: rule on (stage {_paStage}, wait {_paWait}, hands <= {_paD} m, {_paPokes} pokes, "
              f"stage file {_paFile})", flush=True)

    # 2026-10-15 (user: "run the slow-close with button tracking"). BUTTON-SERVO, B1K_T0_BTN=1. The radio's toggle button is
    # the red dot in the centre of the black speaker; the LEFT wrist camera sees it (red blob 8-900 px with a dark ring).
    # Successful presses: 5-20 steps before the toggle the dot sits near pixel (155, 135) of the 224x224 wrist image and
    # grows to ~500 px. Failures (304) saw the dot but the closed fingers came down beside it. While the policy's stage
    # (B1K_STAGE_FILE) is 3 or 4 and R holds: once the dot has been visible for B1K_T0_BTN_WAIT (150) stage steps without a
    # success (the model presses by itself ~150 steps after the dot appears), the L arm is servoed by IK (orientation held):
    # sideways/up-down in the camera plane to put the dot at (B1K_T0_BTN_TX, _TY), and forward along the camera axis
    # while centred (|err| < 30 px) until the dot area >= B1K_T0_BTN_AREA (450) or 8 cm advanced; hold 20, back 2 cm,
    # repeat (6 pokes), then 150 steps of model. L fingers closed. Dot lost 15 steps -> model. Sign self-check: if the pixel
    # error grows 30% over the first 20 servo steps, the lateral mapping is flipped once. Camera + robot state only.
    # Logs "BUTTON-SERVO".
    if os.environ.get("B1K_T0_BTN", "0") == "1":
        _bWait = int(os.environ.get("B1K_T0_BTN_WAIT", "150"))
        _bTx = float(os.environ.get("B1K_T0_BTN_TX", "155")); _bTy = float(os.environ.get("B1K_T0_BTN_TY", "135"))
        _bArea = float(os.environ.get("B1K_T0_BTN_AREA", "450")); _bFile = os.environ.get("B1K_STAGE_FILE", "")
        _orig_eval_step_b = eval_mod.Evaluator.step

        def _b_t(x):
            return th.as_tensor(x.cpu().numpy() if hasattr(x, "cpu") else x, dtype=th.float32)

        def _b_stage():
            try:
                with open(_bFile) as fh:
                    return int(fh.read().split()[1])
            except Exception:
                return -1

        def _b_width(robot, arm):
            gi = _b_t(robot.gripper_control_idx[arm]).long()
            q = _b_t(robot.get_joint_positions())[gi]
            lo, hi = _b_t(robot.joint_lower_limits)[gi], _b_t(robot.joint_upper_limits)[gi]
            return float((2 * (q - lo) / (hi - lo).clamp(min=1e-6) - 1).mean())

        def _b_dot(ev):
            import cv2 as _cv2
            import numpy as _bn
            k = eval_mod.ROBOT_CAMERA_NAMES["R1Pro"]["left_wrist"] + "::rgb"
            if not isinstance(getattr(ev, "obs", None), dict) or k not in ev.obs:
                return None
            im = ev.obs[k]
            im = im.cpu().numpy() if hasattr(im, "cpu") else _bn.asarray(im)
            im = _cv2.resize(_bn.ascontiguousarray(im[..., :3]).astype(_bn.uint8), (224, 224)).astype(int)
            r, g, b = im[..., 0], im[..., 1], im[..., 2]
            red = ((r > 150) & (g < 90) & (b < 90)).astype(_bn.uint8)
            dark = ((r < 70) & (g < 70) & (b < 70)).astype(_bn.uint8)
            n, lab, stats, cent = _cv2.connectedComponentsWithStats(red)
            best = None
            for j in range(1, n):
                if not 8 <= stats[j, 4] <= 900:
                    continue
                m = (lab == j).astype(_bn.uint8)
                ring = _cv2.dilate(m, _bn.ones((15, 15), _bn.uint8)) - _cv2.dilate(m, _bn.ones((5, 5), _bn.uint8))
                fr = float(dark[ring > 0].mean()) if ring.any() else 0.0
                if fr > 0.5 and (best is None or stats[j, 4] > best[2]):
                    best = (float(cent[j][0]), float(cent[j][1]), int(stats[j, 4]))
            return best

        def _b_cam(robot):
            for nm, sn in robot.sensors.items():
                if "left_realsense_link" in nm:
                    p, q = sn.get_position_orientation()
                    return _b_t(T.quat2mat(_b_t(q)))
            return None

        def _b_ik(robot, prev, Rb, v):
            idx = _b_t(robot.arm_control_idx["left"]).long()
            twist = th.cat([Rb.T @ v, th.zeros(3)])
            J = _b_t(robot.get_control_dict()["eef_left_jacobian_relative"])[:, idx]
            q = prev + J.T @ th.linalg.solve(J @ J.T + 0.0025 * th.eye(6), twist)
            lo, hi = _b_t(robot.joint_lower_limits)[idx], _b_t(robot.joint_upper_limits)[idx]
            return th.minimum(th.maximum(q, lo + 0.01), hi - 0.01)

        def _b_step(self):
            pf = self.policy.forward
            robot = self.robot
            s = self.__dict__.setdefault("_bs", {"t": -1})
            t = int(self.env._current_step)
            if t < s["t"] or "vis" not in s:
                s.update(vis=0, ph="model", q=None, n=0, k=0, adv=0.0, lost=0, cool=0, sign=1.0, e0=None, chk=False,
                         takes=0)
            s["t"] = t

            def _gated(*args, **kw):
                a = pf(*args, **kw)
                try:
                    stg = _b_stage()
                    if stg not in (3, 4) or not (-0.95 < _b_width(robot, "right") < 0.9):
                        if s["ph"] != "model":
                            print(f"BUTTON-SERVO: off at step {t} (stage {stg} / R not holding)", flush=True)
                            s.update(ph="model", q=None)
                        return a
                    d = _b_dot(self)
                    if s["ph"] == "model":
                        if d is not None:
                            s["vis"] += 1
                        if s["cool"] > 0:
                            s["cool"] -= 1
                            return a
                        if d is None or s["vis"] < _bWait:
                            return a
                        idx = _b_t(robot.arm_control_idx["left"]).long()
                        s.update(ph="servo", q=_b_t(robot.get_joint_positions())[idx].clone(), n=0, k=1, adv=0.0, lost=0,
                                 e0=None, chk=False)
                        s["takes"] += 1
                        print(f"BUTTON-SERVO: take-over #{s['takes']} at step {t} (dot at ({d[0]:.0f},{d[1]:.0f}) "
                              f"{d[2]} px, seen {s['vis']} press-stage steps)", flush=True)
                    a[14] = -1.0
                    bpos, bq = robot.get_position_orientation()
                    Rb = _b_t(T.quat2mat(_b_t(bq)))
                    Rc = _b_cam(robot)
                    if Rc is None:
                        print("BUTTON-SERVO: no left wrist camera sensor -> off", flush=True)
                        s.update(ph="model", cool=10 ** 9)
                        return a
                    xr, yu, fw = Rc[:, 0], Rc[:, 1], -Rc[:, 2]     # image right, image up, optical axis (USD camera)
                    s["n"] += 1
                    if d is None:
                        s["lost"] += 1
                        if s["lost"] >= 15:
                            print(f"BUTTON-SERVO: dot lost at step {t} -> model for 150 steps", flush=True)
                            s.update(ph="model", q=None, cool=150, vis=0)
                            return a
                        v = th.zeros(3)
                    else:
                        s["lost"] = 0
                        ex, ey = d[0] - _bTx, d[1] - _bTy
                        err = (ex * ex + ey * ey) ** 0.5
                        if s["e0"] is None:
                            s["e0"] = (err, s["n"])
                        elif not s["chk"] and s["n"] - s["e0"][1] >= 20:
                            s["chk"] = True
                            if err > 1.3 * s["e0"][0] and s["e0"][0] > 20:
                                s["sign"] = -s["sign"]
                                print(f"BUTTON-SERVO: error grew {s['e0'][0]:.0f} -> {err:.0f} px -> lateral sign flipped",
                                      flush=True)
                        g = 0.00025 * s["sign"]
                        v = g * ex * xr - g * ey * yu
                        if s["ph"] == "servo" and err < 30:
                            v = v + 0.0015 * fw
                            s["adv"] += 0.0015
                            if d[2] >= _bArea or s["adv"] >= 0.08:
                                print(f"BUTTON-SERVO: poke #{s['k']} contact at step {t} (dot {d[2]} px, err {err:.0f} px,"
                                      f" advanced {100 * s['adv']:.1f} cm) -> hold", flush=True)
                                s.update(ph="hold", n=0)
                        if s["n"] % 30 == 0:
                            print(f"BUTTON-SERVO: step {t} {s['ph']} dot ({d[0]:.0f},{d[1]:.0f}) {d[2]} px err {err:.0f} px",
                                  flush=True)
                    if s["ph"] == "hold":
                        v = th.zeros(3)
                        if s["n"] >= 20:
                            s.update(ph="back", n=0)
                    elif s["ph"] == "back":
                        v = -0.0015 * fw
                        if s["n"] >= 14:
                            if s["k"] >= 6:
                                print(f"BUTTON-SERVO: 6 pokes done at step {t} -> model for 150 steps", flush=True)
                                s.update(ph="model", q=None, cool=150, vis=0)
                                return a
                            s["k"] += 1
                            s.update(ph="servo", n=0, adv=0.0)
                    vn = float(v.norm())
                    if vn > 0.004:
                        v = v * (0.004 / vn)
                    s["q"] = _b_ik(robot, s["q"], Rb, v)
                    a[7:14] = s["q"].to(a.dtype) if hasattr(a, "cpu") else s["q"].numpy()
                except Exception as e:
                    if not s.get("err"):
                        s["err"] = True
                        print(f"BUTTON-SERVO: error {e}", flush=True)
                return a

            self.policy.forward = _gated
            try:
                return _orig_eval_step_b(self)
            finally:
                self.policy.forward = pf

        eval_mod.Evaluator.step = _b_step
        print(f"BUTTON-SERVO: rule on (wait {_bWait}, target ({_bTx:.0f},{_bTy:.0f}), contact area {_bArea:.0f} px, "
              f"stage file {_bFile})", flush=True)

    # 2026-10-12 (user, task 63): WALL-HOLD / WALL-TWIST. Close-freeze run 0abd041 inst 2 carried the smoke detector to
    # 8.4 cm / 102 deg from the nail (hand 1.64 m up, detector 7 cm below the nail) and then the model opened the gripper.
    # Robot-state only (own kinematics + finger width, no object poses):
    #   held      - a hand commanded closed with finger width inside (-0.95, 0.97) for >= 20 steps
    #   WALL-HOLD (B1K_WALL_HOLD=<target m>, e.g. 1.68): while a hand is held, an open command is replaced by close if
    #             the eef is below B1K_WALL_MIN (1.45 m above the base) = no drops on the way (cap B1K_WALL_HOLD_MAX 3000
    #             blocked steps). An open command at/above WALL_MIN = the model thinks it is at the nail: base stopped,
    #             torso held, gripper closed, the hand is lifted 1 mm/step up to <target> (or until it stalls: < 3 mm in
    #             40 steps, or 150 steps); then the gripper opens for 60 steps and the model is in charge again.
    #   WALL-TWIST (B1K_WALL_TWIST=1, needs WALL_HOLD): after the lift, instead of opening, sweep the hand's roll about the
    #             robot's forward (horizontal) axis = the wall normal: for each offset in B1K_WALL_ANGLES (+-15 ... 180):
    #             back off 2 cm, turn (<= 1 deg/step, timeout 200), press up to 3.5 cm toward the wall (stops when it moves
    #             < 2 mm in 10 steps), dwell 15, pull back 1.5 cm: if the hand moves < 5 mm it is locked to the nail ->
    #             open 60 steps, model in charge. Ends with open after the last angle. Once per episode.
    # Wraps policy.forward like CLOSE-BELOW. Logs "WALL-HOLD:" / "WALL-TWIST:".
    _wh = os.environ.get("B1K_WALL_HOLD", "")
    if _wh:
        import math as _wm
        _whZ = float(_wh); _whMin = float(os.environ.get("B1K_WALL_MIN", "1.45"))
        _whMax = int(os.environ.get("B1K_WALL_HOLD_MAX", "3000"))
        _wtOn = os.environ.get("B1K_WALL_TWIST", "0") == "1"
        # 2nd trigger (01:55, clean t63 inst 2 held the detector 2.7 cm from the nail, 68 deg off, from ~2000 to the end
        # without ever opening): a held hand at/above WALL_MIN that stays within 3 cm for B1K_WALL_STILL (300) steps
        _whStill = int(os.environ.get("B1K_WALL_STILL", "300" if _wtOn else "0"))
        # v3 (06:10, WALLTWIST2: the sweep ran but tried only 10/23 angles before the episode limit (1 deg/step, back-and-forth
        # order) and the hand drifted 30 cm sideways along the wall): B1K_WALL_FAST=1 - turn B1K_WALL_TURN deg/step (3),
        # one-directional order (-15..-180, then back through 0 to +15..+165), shorter cycle (back 2 cm @2 mm, press <= 4 cm
        # @2 mm stopping on < 2 mm progress in 5 steps, dwell 10, pull 1.5 cm @1.5 mm), and the hand is held on the sweep-start
        # point across the wall (lateral + vertical, <= 3 mm/step).
        _wtFast = os.environ.get("B1K_WALL_FAST", "0") == "1"
        _wtTurn = float(os.environ.get("B1K_WALL_TURN", "3"))
        # v6 (09:00, ORACLE4 q=1.0: aligned AND pressed 4 cm past the nail point -> contact -> attached; ORACLE3b pressed only
        # 1 cm past and touched nothing but its own fingers): B1K_WALL_PRESS=m - fast-mode press goes up to m (0.03) past the
        # sweep start; on contact (< 1 mm per 5 steps after 10 steps) a "seat" phase keeps commanding 2 mm/step further for
        # B1K_WALL_SEAT steps (8 = ~1.6 cm of wind-up = pressing force), then dwell / pull as before.
        _wtPress = float(os.environ.get("B1K_WALL_PRESS", "0.03"))
        _wtSeat = int(os.environ.get("B1K_WALL_SEAT", "0"))
        # v7 (09:22, WALLTWIST5b: the pull was really blocked at -15 deg but the detector was only SNAGGED (1.7 cm / 46 deg,
        # not attached) -> released -> dropped; ORACLE4: a real attachment ends the episode as a success at once).
        # B1K_WALL_NOLOCK=1: never release on a blocked pull - log it as a snag and go on to the next angle (the anchored
        # "back" phase pulls the hand free). The gripper only opens after the last angle.
        _wtNoLock = os.environ.get("B1K_WALL_NOLOCK", "0") == "1"
        # v8 (09:55, WALLTWIST7: the model arrived at 8 cm / 7-10 deg but the sweep turned away before any press, and every
        # press stalled ~6 cm short of the nail with the hand 0.82 m from the base = arm at full reach; ORACLE4 attached with
        # the base crept in). B1K_WALL_ZERO_FIRST=1: try 0 deg (the model's own orientation) first. B1K_WALL_REACH=m: during
        # the press, if the hand is more than m (horizontal) from the base centre, drive the base forward (0.08, robot frame).
        _wtZero = os.environ.get("B1K_WALL_ZERO_FIRST", "0") == "1"
        # v11 (13:08, WALLTWIST10: the model opened at the wall with the hand at 1.777 m and the whole sweep pressed ~10 cm
        # ABOVE the nail; ORACLE4 attached with the hand at ~1.70 m): B1K_WALL_EXACT=1 - the pre-sweep "lift" moves the hand
        # to exactly B1K_WALL_HOLD (up OR down, 1 mm/step, done within 5 mm or on a 40-step stall / 150 steps).
        _whExact = os.environ.get("B1K_WALL_EXACT", "0") == "1"
        _wtReach = float(os.environ.get("B1K_WALL_REACH", "0"))
        _wtAng = [float(x) for x in os.environ.get(
            "B1K_WALL_ANGLES", "15,-15,30,-30,45,-45,60,-60,75,-75,90,-90,105,-105,120,-120,135,-135,150,-150,165,-165,180"
        ).split(",")]
        if _wtFast and "B1K_WALL_ANGLES" not in os.environ:
            _wtAng = [-15.0 * k for k in range(1, 13)] + [15.0 * k for k in range(1, 12)]
        if _wtZero:
            _wtAng = [0.0] + _wtAng
        # v12 (16:20, RULESPASS inst 1: detector at the wall, right height, ~10 deg off, but 7 cm SIDEWAYS of the nail -
        # the angle-only sweep pressed at the same spot 15 times): B1K_WALL_LAT="0,-0.04,0.04,..." = sideways offsets (m,
        # along the wall, from where the model opened) tried at each of B1K_WALL_LAT_ANG first, then the angle sweep at 0
        _wtLat = [float(x) for x in os.environ.get("B1K_WALL_LAT", "").split(",") if x.strip()]
        _wtLatAng = [float(x) for x in os.environ.get("B1K_WALL_LAT_ANG", "0").split(",") if x.strip()]
        _wtPairs = [(g, d) for g in _wtLatAng for d in _wtLat] if _wtLat else []
        _wtPairs += [(g, 0.0) for g in _wtAng if (g, 0.0) not in _wtPairs]
        # v13 (17:15, v12 inst 1: detector 2.5 cm from the nail but ~30 deg off about the wall normal; the 21 sideways tries
        # came first, didn't move the detector much and sagged the hand 30 cm before the +-30 presses): B1K_WALL_LAT_AFTER=30
        # = small angles (|a| <= 30) first, then the sideways tries (at 0 deg only), then the rest of the angle sweep
        # DOT-ALIGN (2026-10-13, user idea): the smoke detector has a red dot that the LEFT WRIST camera (the policy's own
        # input image) sees while carrying. Its angle in the 224x224 wrist image around the grip point (95, 120) predicts the
        # roll about the wall normal still needed at the wall: needed ~= dot_angle - B1K_WALL_DOT_OFF (177 deg; fit on 7
        # arrivals, residual ~+-12 deg). B1K_WALL_DOT=1: the sweep's first presses are needed+(0,-10,+10,-20,+20) deg,
        # then the normal sweep. Uses only the onboard camera image (no sim poses). Logs "DOT-ALIGN".
        _wtDot = os.environ.get("B1K_WALL_DOT", "0") == "1"
        _wtDotOff = float(os.environ.get("B1K_WALL_DOT_OFF", "177"))
        # YAW-LOCK (05:30, user: "the arm was perfect but the base moved wrong"; DOTALIGN inst 1: at 2800 the hand was 4 cm
        # from the nail with the base square to the wall, then the MODEL turned the base 11 deg (2800-2880) and the 0.86 m
        # arm swung 16 cm sideways; the sweep then anchored there and crept along the turned heading). B1K_WALL_YAWLOCK=1:
        # while the left hand carries the detector at >= B1K_WALL_MIN, the model's base turn command is zeroed and its
        # forward/sideways commands are capped at B1K_WALL_YAWLOCK_V (0.05). Robot state only. Logs "YAW-LOCK".
        _wtYawLock = os.environ.get("B1K_WALL_YAWLOCK", "0") == "1"
        _wtYawV = float(os.environ.get("B1K_WALL_YAWLOCK_V", "0.05"))
        # B1K_WALL_YAWLOCK_VX (2026-10-13, user: 301 "takes far too long inching towards the wall"): separate cap for the
        # FORWARD speed. With the 0.05 cap the model's 0.45 forward bursts were clipped 9x and 301 crawled the last 1.5 m
        # at ~0.017 m/s (1800 steps). The yaw swing is what moved the hand sideways, so forward can stay uncapped.
        _wtYawVX = float(os.environ.get("B1K_WALL_YAWLOCK_VX", str(_wtYawV)))
        # B1K_WALL_DOT_RIGHT=1 (2026-10-14, public 310: the RIGHT hand carried the detector, DOT-ALIGN/YAW-LOCK only
        # looked at the left, and it sat 2-4 cm from the nail but 95-110 deg off): YAW-LOCK and the red-dot reads follow
        # whichever hand carries, reading that hand's wrist camera. Calibration (diag/dotcal, 3 right-hand arrivals):
        # the left-hand formula (grip point (95,120), need = dot - B1K_WALL_DOT_OFF) fits with residuals +0/-2/+26 deg.
        _wtDotR = os.environ.get("B1K_WALL_DOT_RIGHT", "0") == "1"
        # 03:55 (DOTALIGN inst 1: angle came to 11 deg but the detector sat 10-13 cm sideways of the nail): after the 5 dot
        # angles, sideways tries at the dot angle (B1K_WALL_DOT_LAT, m); those back off B1K_WALL_LAT_BACK (5 cm) first
        _wtDotLat = [float(x) for x in os.environ.get("B1K_WALL_DOT_LAT", "").split(",") if x.strip()]
        # B1K_WALL_DOT_OFFS (2026-10-13, public 304 sat at 0.4 cm / 15.0 deg = just outside the < 15 deg attach limit):
        # angle offsets tried around the dot estimate, default the original 0,-10,10,-20,20 (e.g. 0,-6,6,-12,12,-18,18,-24,24)
        _wtDotOffs = [float(x) for x in os.environ.get("B1K_WALL_DOT_OFFS", "0,-10,10,-20,20").split(",") if x.strip()]
        _wtLatBack = float(os.environ.get("B1K_WALL_LAT_BACK", "0.05"))
        # B1K_WALL_CREEP_PRESS=1 (2026-10-13, public test: ~half the pull-backs moved 0 cm because the reach creep kept
        # driving the base forward at 0.08 during back/pull/dwell, pinning the detector to the wall = no new contact):
        # base forward creep only in the turn and press phases.
        _wtCreepPress = os.environ.get("B1K_WALL_CREEP_PRESS", "0") == "1"
        # B1K_WALL_PULL_BASE=1 (2026-10-13, CREEP_PRESS pass: pulls still moved 0 cm - at the wall the arm is stretched
        # 0.85-0.9 m, near-singular, so the arm-only pull barely retracts): during the pull also drive the base backward
        # (-0.06) and keep pulling until the hand is >= 1.5 cm back along the wall normal (max 40 steps).
        _wtPullBase = os.environ.get("B1K_WALL_PULL_BASE", "0") == "1"
        # B1K_WALL_PRESS_CREEP=v (2026-10-13, PULLBASE carriers 306/308: every press stalled 8-17 cm short of the wall with the
        # hand only ~0.70 m from the base, under the 0.78 reach-creep threshold, so the base never closed the gap): during
        # the press phase drive the base forward at v regardless of reach; the wall stops the detector, which ends the press.
        _wtPressCreep = float(os.environ.get("B1K_WALL_PRESS_CREEP", "0"))
        # ALIGN-EARLY (2026-10-13, user: "turn the dot to pi, raise the arm to the correct height and then give the base to
        # the model"): B1K_WALL_EARLY=1. Once the carried left hand passes B1K_WALL_EARLY_Z (1.30 m above the base) and the
        # wrist camera has >= 3 red-dot reads: lift to the wall height (B1K_WALL_HOLD), roll the hand about the base heading
        # by (dot angle - B1K_WALL_DOT_OFF), then FREEZE arm + torso and let the model drive the base (yaw zeroed, speed
        # capped). On contact (hand stops advancing while the model drives forward), a model open, or B1K_WALL_EARLY_MAX
        # steps, start the WALL-TWIST sweep around the aligned roll (B1K_WALL_DOT_OFFS first). Wrist camera + arm state only.
        _weOn = os.environ.get("B1K_WALL_EARLY", "0") == "1"
        _weZ = float(os.environ.get("B1K_WALL_EARLY_Z", "1.30"))
        _weMax = int(os.environ.get("B1K_WALL_EARLY_MAX", "2000"))
        _weYaw = float(os.environ.get("B1K_WALL_EARLY_YAW", "0.15"))

        def _w_dot(ev, arm="left"):
            import cv2 as _cv2
            import numpy as _dn
            k = eval_mod.ROBOT_CAMERA_NAMES["R1Pro"][f"{arm}_wrist"] + "::rgb"
            if not isinstance(getattr(ev, "obs", None), dict) or k not in ev.obs:
                return None
            im = ev.obs[k]
            im = im.cpu().numpy() if hasattr(im, "cpu") else _dn.asarray(im)
            im = _cv2.resize(_dn.ascontiguousarray(im[..., :3]).astype(_dn.uint8), (224, 224)).astype(int)
            m = (im[..., 0] > 140) & (im[..., 1] < 90) & (im[..., 2] < 90)
            ys, xs = _dn.nonzero(m)
            if len(xs) < 4:
                return None
            return float(_dn.degrees(_dn.arctan2(-(ys.mean() - 120.0), xs.mean() - 95.0)) % 360.0), int(len(xs))

        _wtLatAfter = float(os.environ.get("B1K_WALL_LAT_AFTER", "-1"))
        if _wtLat and _wtLatAfter >= 0:
            _sm = [(g, 0.0) for g in _wtAng if abs(g) <= _wtLatAfter]
            _wtPairs = _sm + [(0.0, d) for d in _wtLat if d != 0.0] + [(g, 0.0) for g in _wtAng if abs(g) > _wtLatAfter]
        _orig_eval_step_wh = eval_mod.Evaluator.step
        _WG = {"left": 14, "right": 22}

        def _w_t(x):
            return th.as_tensor(x.cpu().numpy() if hasattr(x, "cpu") else x, dtype=th.float32)

        def _w_set(a, sl, v):
            v = _w_t(v)
            a[sl] = v.to(a.dtype) if hasattr(a, "cpu") else v.numpy()

        def _w_width(robot, arm):
            gi = _w_t(robot.gripper_control_idx[arm]).long()
            q = _w_t(robot.get_joint_positions())[gi]
            lo, hi = _w_t(robot.joint_lower_limits)[gi], _w_t(robot.joint_upper_limits)[gi]
            return float((2 * (q - lo) / (hi - lo).clamp(min=1e-6) - 1).mean())

        def _w_ik(robot, arm, prev, Rb, v_lin, om):
            idx = _w_t(robot.arm_control_idx[arm]).long()
            twist = th.cat([Rb.T @ v_lin, Rb.T @ om])
            J = _w_t(robot.get_control_dict()[f"eef_{arm}_jacobian_relative"])[:, idx]
            q = prev + J.T @ th.linalg.solve(J @ J.T + 0.0025 * th.eye(6), twist)
            lo, hi = _w_t(robot.joint_lower_limits)[idx], _w_t(robot.joint_upper_limits)[idx]
            return th.minimum(th.maximum(q, lo + 0.01), hi - 0.01)

        def _w_roll(robot, arm, f, col):
            # angle of eef axis <col> in the plane perpendicular to the forward axis f (basis: world up, f x up)
            Re = _w_t(T.quat2mat(_w_t(robot.get_eef_orientation(arm=arm))))
            up = th.tensor([0.0, 0.0, 1.0]); side = th.linalg.cross(f, up)
            x = Re[:, col]
            return _wm.atan2(float(x @ side), float(x @ up))

        def _wh_step(self):
            pf = self.policy.forward
            robot = self.robot
            t = int(self.env._current_step)
            s = getattr(self, "_whs", None)
            if s is None or t < s["t"]:
                s = self._whs = {"t": t, "held": {"left": 0, "right": 0}, "mode": "idle", "arm": None, "n": 0,
                                 "blocked": 0, "twisted": False, "q": None, "torso": None,
                                 "z_close": {}, "carry": {}, "bh": {}}
                print(f"WALL-HOLD: episode start (step {t})", flush=True)
            s["t"] = t

            def _gated(*args, **kw):
                a = pf(*args, **kw)
                try:
                    _wh_apply(self, robot, s, a, t)
                except Exception as e:
                    print(f"WALL-HOLD: error {e}", flush=True)
                return a

            self.policy.forward = _gated
            try:
                return _orig_eval_step_wh(self)
            finally:
                self.policy.forward = pf

        # GRASP-DESCEND (2026-10-12 20:15, user: t63 pickups are "grasping luck"; at the closes the left eef sat 1.5-4.5 cm
        # above the detector centre -> rim pinches / air closes). Robot-state only: when the model commands a close on an
        # OPEN hand that is low (eef < B1K_GD_MAXZ above the base) and not carrying, keep the fingers open and lower the hand
        # (1 mm/step, base still, torso held) until it stalls on something (contact) or B1K_GD_DEPTH (3 cm), then close
        # with the arm held for B1K_GD_CLOSE (25) steps and hand back. Cooldown B1K_GD_COOL (60) steps per arm.
        _gdOn = os.environ.get("B1K_GRASP_DESCEND", "0") == "1"
        _gdMaxZ = float(os.environ.get("B1K_GD_MAXZ", "0.70"))
        _gdDepth = float(os.environ.get("B1K_GD_DEPTH", "0.03"))
        _gdClose = int(os.environ.get("B1K_GD_CLOSE", "25"))
        _gdCool = int(os.environ.get("B1K_GD_COOL", "60"))

        # PICK-DOT (2026-10-13, user: "add an align rule when picking up so the range of motion is not limited"): with
        # B1K_GD_DOT=1, before the GRASP-DESCEND lowering, spin the LEFT gripper about the vertical (hand held in place,
        # fingers open) until the detector's red dot sits at B1K_GD_DOT_TARGET (177 deg) around the detector centre in the
        # left wrist image (centre = centroid of the dark vent stripes in the middle of the image). Closed loop: the turn
        # direction flips once if the error grows; stops within 8 deg, on a 30-step stall (joint limit), after 160 steps,
        # or when the dot is lost for 20 steps. Onboard camera only. Logs "PICK-DOT".
        _gdDot = os.environ.get("B1K_GD_DOT", "0") == "1"
        _gdDotT = float(os.environ.get("B1K_GD_DOT_TARGET", "177"))
        # B1K_GD_HOLDXY=1 (2026-10-13, public test: the PICK-DOT turn moved the hand 6-11 cm off the detector, e.g. 302
        # 2.2 -> 9.1 cm, 304 0.8 -> 10.8 cm; successful grasps stayed within ~3 cm): during the turn and the lowering, servo
        # the eef back to where it was when the close was triggered (position kept in the base frame = arm kinematics only).
        _gdHold = os.environ.get("B1K_GD_HOLDXY", "0") == "1"

        def _gd_hold(robot, arm, d, Rb, bpos, ep, z=True):
            if not _gdHold or d.get("p0") is None:
                return th.zeros(3)
            v = (bpos + Rb @ d["p0"]) - ep
            if not z:
                v[2] = 0.0
            v = 0.3 * v
            n = float(v.norm())
            return v * (0.003 / n) if n > 0.003 else v

        def _pick_dot(ev):
            import cv2 as _cv2
            import numpy as _dn
            k = eval_mod.ROBOT_CAMERA_NAMES["R1Pro"]["left_wrist"] + "::rgb"
            if not isinstance(getattr(ev, "obs", None), dict) or k not in ev.obs:
                return None
            im = ev.obs[k]
            im = im.cpu().numpy() if hasattr(im, "cpu") else _dn.asarray(im)
            im = _cv2.resize(_dn.ascontiguousarray(im[..., :3]).astype(_dn.uint8), (224, 224)).astype(int)
            red = (im[..., 0] > 140) & (im[..., 1] < 90) & (im[..., 2] < 90)
            ys, xs = _dn.nonzero(red)
            if len(xs) < 4:
                return None
            sub = im[30:200, 40:184]
            dk = sub.max(axis=2) < 70
            yy, xx = _dn.nonzero(dk)
            if len(xx) < 30:
                return None
            cx, cy = xx.mean() + 40.0, yy.mean() + 30.0
            return float(_dn.degrees(_dn.arctan2(-(ys.mean() - cy), xs.mean() - cx)) % 360.0), cx, cy

        def _gd_apply(robot, s, a, t, Rb, bpos, zero, ev=None):
            gd = s.setdefault("gd", {})
            for arm, g in _WG.items():
                d = gd.setdefault(arm, {"ph": None, "cool": -999, "nev": 0})
                sl = slice(7, 14) if arm == "left" else slice(15, 22)
                ep = _w_t(robot.get_eef_position(arm=arm))
                hz = float((Rb.T @ (ep - bpos))[2])
                if d["ph"] is None:
                    if (float(a[g]) < 0 and _w_width(robot, arm) > 0.5 and hz < _gdMaxZ
                            and not s["carry"].get(arm, False) and t - d["cool"] > _gdCool):
                        jq = _w_t(robot.get_joint_positions())
                        d.update(ph="down", n=0, z0=hz, zl=hz, q=jq[_w_t(robot.arm_control_idx[arm]).long()].clone(),
                                 torso=jq[_w_t(robot.trunk_control_idx).long()].clone(), p0=Rb.T @ (ep - bpos))
                        if _gdDot and arm == "left":
                            d.update(ph="turn", dir=0.0, lost=0, e8=None, flipped=False, last=None, stallref=None)
                        d["nev"] += 1
                        if d["nev"] <= 30:
                            print(f"GRASP-DESCEND: {arm} close at step {t} (eef {hz:.3f} m above the base) -> lowering "
                                  f"first (#{d['nev']})", flush=True)
                    else:
                        continue
                d["n"] += 1
                a[0] = 0.0; a[1] = 0.0; a[2] = 0.0
                _w_set(a, slice(3, 7), d["torso"])
                if d["ph"] == "turn":
                    a[g] = 1.0
                    pdv = None
                    try:
                        pdv = _pick_dot(ev)
                    except Exception as e:
                        if not s.get("pderr"):
                            s["pderr"] = True; print(f"PICK-DOT: read error {e}", flush=True)
                    done = None
                    if pdv is None:
                        d["lost"] += 1
                        wz = 0.0
                        if d["lost"] >= 20:
                            done = "dot not seen"
                    else:
                        d["lost"] = 0
                        err = (pdv[0] - _gdDotT + 180.0) % 360.0 - 180.0
                        if d["n"] == 1 or d["last"] is None:
                            d["err0"] = err
                        if d["dir"] == 0.0:
                            d["dir"] = 1.0 if err > 0 else -1.0
                        if d["n"] % 8 == 0:
                            if d["e8"] is not None and abs(err) > abs(d["e8"]) + 2.0 and not d["flipped"]:
                                d["dir"] = -d["dir"]; d["flipped"] = True
                            d["e8"] = err
                        if d["n"] % 30 == 0:
                            if d["stallref"] is not None and abs(err - d["stallref"]) < 3.0 and d["n"] >= 30:
                                done = f"stalled (joint limit?) at err {err:+.0f}"
                            d["stallref"] = err
                        if abs(err) < 8.0:
                            done = f"dot at {pdv[0]:.0f} deg (err {err:+.0f})"
                        wz = d["dir"] * _wm.radians(min(1.5, max(0.4, abs(err) * 0.05)))
                        d["last"] = err
                    if d["n"] >= 160 and done is None:
                        done = "160 steps"
                    if done is not None:
                        if d["nev"] <= 30:
                            off = float((bpos + Rb @ d["p0"] - ep)[:2].norm()) * 100 if d.get("p0") is not None else -1
                            print(f"PICK-DOT: left turn done at step {t} ({done}; started at err "
                                  f"{d.get('err0', float('nan')):+.0f}; hand {off:.1f} cm from its start)", flush=True)
                        jq = _w_t(robot.get_joint_positions())
                        d.update(ph="down", n=0, q=jq[_w_t(robot.arm_control_idx[arm]).long()].clone())
                    else:
                        d["q"] = _w_ik(robot, arm, d["q"], Rb, _gd_hold(robot, arm, d, Rb, bpos, ep),
                                       th.tensor([0.0, 0.0, float(wz)]))
                elif d["ph"] == "down":
                    a[g] = 1.0
                    d["q"] = _w_ik(robot, arm, d["q"], Rb,
                                   th.tensor([0.0, 0.0, -0.001]) + _gd_hold(robot, arm, d, Rb, bpos, ep, z=False), zero)
                    stall = False
                    if d["n"] % 4 == 0:
                        stall = d["n"] >= 8 and (d["zl"] - hz) < 0.001
                        d["zl"] = hz
                    if stall or d["z0"] - hz >= _gdDepth or d["n"] >= 45:
                        if d["nev"] <= 30:
                            print(f"GRASP-DESCEND: {arm} lowered {100 * (d['z0'] - hz):.1f} cm"
                                  f"{' (stopped on contact)' if stall else ''} at step {t} -> closing", flush=True)
                        jq = _w_t(robot.get_joint_positions())
                        d.update(ph="close", n=0, q=jq[_w_t(robot.arm_control_idx[arm]).long()].clone())
                elif d["ph"] == "close":
                    a[g] = -1.0
                    if d["n"] >= _gdClose:
                        if d["nev"] <= 30:
                            print(f"GRASP-DESCEND: {arm} closed at step {t} (width {_w_width(robot, arm):+.2f}) -> model",
                                  flush=True)
                        d["ph"] = None; d["cool"] = t
                        continue
                _w_set(a, sl, d["q"])
                return

        def _wh_apply(self, robot, s, a, t):
            bpos, bq = robot.get_position_orientation()
            Rb = _w_t(T.quat2mat(_w_t(bq))); bpos = _w_t(bpos)
            f = Rb[:, 0].clone(); f[2] = 0.0; f = f / f.norm().clamp(min=1e-6)
            zero = th.zeros(3)
            if _gdOn and s["mode"] == "idle":
                _gd_apply(robot, s, a, t, Rb, bpos, zero, self)
            ca = "left" if s["carry"].get("left", False) else \
                ("right" if _wtDotR and s["carry"].get("right", False) else None)
            if _wtYawLock and s["mode"] == "idle" and ca:
                hzl = float((Rb.T @ (_w_t(robot.get_eef_position(arm=ca)) - bpos))[2])
                if hzl >= _whMin:
                    if abs(float(a[2])) > 1e-4 or abs(float(a[0])) > _wtYawVX or abs(float(a[1])) > _wtYawV:
                        s["yl"] = s.get("yl", 0) + 1
                        if s["yl"] in (1, 50) or s["yl"] % 500 == 0:
                            print(f"YAW-LOCK: base turn {float(a[2]):+.3f} zeroed at step {t} (hand {hzl:.2f} m, #{s['yl']})",
                                  flush=True)
                    a[2] = 0.0
                    a[0] = max(-_wtYawVX, min(_wtYawVX, float(a[0]))); a[1] = max(-_wtYawV, min(_wtYawV, float(a[1])))
            if _wtDot and s["mode"] in ("idle", "lift") and ca and t % 10 == 0:
                if s.get("dot_arm") != ca:
                    s["dot_arm"] = ca; s["dots"] = []
                try:
                    dd = _w_dot(self, ca)
                except Exception as e:
                    dd = None
                    if not s.get("doterr"):
                        s["doterr"] = True; print(f"DOT-ALIGN: read error {e}", flush=True)
                if dd is not None:
                    s.setdefault("dots", []).append((t, dd[0], dd[1])); del s["dots"][:-15]
            if _weOn and s["mode"] == "idle" and s["carry"].get("left", False) and not s.get("early_done"):
                hzl = float((Rb.T @ (_w_t(robot.get_eef_position(arm="left")) - bpos))[2])
                if hzl >= _weZ:
                    ds = [x[1] for x in s.get("dots", [])[-10:]]
                    if len(ds) >= 3:
                        cx = sum(_wm.cos(_wm.radians(x)) for x in ds); cy = sum(_wm.sin(_wm.radians(x)) for x in ds)
                        dang = _wm.degrees(_wm.atan2(cy, cx)) % 360.0
                        need = (dang - _wtDotOff + 180.0) % 360.0 - 180.0
                        jq = _w_t(robot.get_joint_positions())
                        Re = _w_t(T.quat2mat(_w_t(robot.get_eef_orientation(arm="left"))))
                        col = int(th.argmin(th.abs(Re.T @ f)))
                        s.pop("rp", None)
                        s.update(mode="early", arm="left", ph="lift", n=0, early_done=True, need=need, ru=0.0,
                                 q=jq[_w_t(robot.arm_control_idx["left"]).long()].clone(),
                                 torso=jq[_w_t(robot.trunk_control_idx).long()].clone(), f=f.clone(), col=col,
                                 r0=_w_roll(robot, "left", f, col), z0=hzl, zl=hzl)
                        print(f"ALIGN-EARLY: start at step {t} (hand {hzl:.2f} m; red dot at {dang:.0f} deg, {len(ds)} reads) "
                              f"-> lift to {_whZ:.2f}, roll {need:+.0f} deg, then the model drives", flush=True)
                    elif hzl >= _whMin:
                        s["early_done"] = True
                        print(f"ALIGN-EARLY: no red-dot reads by {hzl:.2f} m at step {t} -> normal rules", flush=True)
            if s["mode"] == "early":
                fw = s["f"]; g = _WG["left"]; mg = float(a[g])
                ep = _w_t(robot.get_eef_position(arm="left")); ez = float((Rb.T @ (ep - bpos))[2])
                s["n"] += 1
                _w_set(a, slice(3, 7), s["torso"])
                a[g] = -1.0
                if s["ph"] in ("lift", "roll"):
                    a[0] = 0.0; a[1] = 0.0; a[2] = 0.0
                    dz = max(-0.002, min(0.002, 0.5 * (_whZ - ez)))
                    if s["ph"] == "lift":
                        s["q"] = _w_ik(robot, "left", s["q"], Rb, th.tensor([0.0, 0.0, dz]), zero)
                        if abs(ez - _whZ) < 0.005 or s["n"] >= 150:
                            print(f"ALIGN-EARLY: lift done at step {t} ({s['z0']:.3f} -> {ez:.3f} m)", flush=True)
                            s.update(ph="roll", n=0, err20=9.0)
                    else:
                        rn = _w_roll(robot, "left", fw, s["col"])
                        s["ru"] = s["ru"] + ((rn - s.get("rp", s["r0"]) + _wm.pi) % (2 * _wm.pi) - _wm.pi)
                        s["rp"] = rn
                        err = _wm.radians(s["need"]) - s["ru"]
                        rate = _wm.radians(_wtTurn)
                        s["q"] = _w_ik(robot, "left", s["q"], Rb, th.tensor([0.0, 0.0, dz]), max(-rate, min(rate, err)) * fw)
                        stalled = False
                        if s["n"] % 20 == 0:
                            stalled = abs(err) > abs(s["err20"]) - _wm.radians(2); s["err20"] = err
                        if abs(err) < _wm.radians(2) or s["n"] >= 150 or (stalled and s["n"] >= 40):
                            print(f"ALIGN-EARLY: roll done at step {t} ({_wm.degrees(s['ru']):+.0f} of {s['need']:+.0f} deg"
                                  f"{', stalled' if stalled else ''}; hand {ez:.3f} m) -> model drives the base", flush=True)
                            jq = _w_t(robot.get_joint_positions())
                            s.update(ph="drive", n=0, q=jq[_w_t(robot.arm_control_idx["left"]).long()].clone(),
                                     pl=ep.clone(), slow=0, opn=0)
                else:
                    # the arm is frozen to the base, so a base turn keeps the roll about the arm's forward axis: let the
                    # model turn (capped, B1K_WALL_EARLY_YAW rad/s) so it can still square up to the wall
                    a[2] = max(-_weYaw, min(_weYaw, float(a[2])))
                    a[0] = max(-_wtYawVX, min(_wtYawVX, float(a[0]))); a[1] = max(-_wtYawV, min(_wtYawV, float(a[1])))
                    s["opn"] = s["opn"] + 1 if mg >= 0 else 0
                    hit = None
                    if s["n"] % 20 == 0:
                        adv = float((ep - s["pl"]) @ fw); s["pl"] = ep.clone()
                        s["slow"] = s["slow"] + 1 if (adv < 0.003 and float(a[0]) > 0.01) else 0
                        if s["slow"] >= 2 and s["n"] >= 60:
                            hit = "hand stopped advancing (contact)"
                    if s["opn"] >= 10:
                        hit = "model opened"
                    if s["n"] >= _weMax:
                        hit = f"{_weMax} steps"
                    if hit is not None:
                        Re = _w_t(T.quat2mat(_w_t(robot.get_eef_orientation(arm="left"))))
                        col = int(th.argmin(th.abs(Re.T @ f)))
                        a0 = ep.clone()
                        sv = th.stack([-f[1], f[0], th.zeros(())]); sv = sv / th.clamp(th.linalg.norm(sv), min=1e-6)
                        first = [(o, 0.0) for o in _wtDotOffs] + [(0.0, dl) for dl in _wtDotLat]
                        s.pop("rp", None)
                        s.update(mode="twist", phase="back", k=0, n=0, f=f, col=col, r0=_w_roll(robot, "left", f, col),
                                 ru=0.0, anchor=a0.clone(), a0=a0, sv=sv, twisted=True, err20=9.0,
                                 pairs=first + [p for p in _wtPairs if all(abs(p[0] - q_[0]) > 5 for q_ in first)])
                        print(f"ALIGN-EARLY: at the wall at step {t} ({hit}; hand {ez:.3f} m) -> WALL-TWIST sweep around the "
                              f"aligned roll ({len(s['pairs'])} tries)", flush=True)
                        s["q"] = _w_t(robot.get_joint_positions())[_w_t(robot.arm_control_idx["left"]).long()].clone()
                _w_set(a, slice(7, 14), s["q"])
                return
            if s["mode"] == "idle":
                for arm, g in _WG.items():
                    w = _w_width(robot, arm)
                    still = False
                    if s["held"][arm] >= 20 and float(a[g]) < 0 and _whStill > 0 and s["carry"].get(arm, False):
                        ep0 = _w_t(robot.get_eef_position(arm=arm))
                        hz = float((Rb.T @ (ep0 - bpos))[2])
                        hist = s.setdefault("hist", {}).setdefault(arm, [])
                        if t % 10 == 0:
                            hist.append((t, ep0.clone(), hz)); del hist[:-(_whStill // 10 + 1)]
                        if hz >= _whMin and len(hist) > _whStill // 10 and hist[0][0] <= t - _whStill and \
                                all(float((h[1] - ep0).norm()) < 0.03 and h[2] >= _whMin for h in hist):
                            still = True
                            print(f"WALL-HOLD: {arm} holding still at the wall for {_whStill} steps at step {t} "
                                  f"(eef {hz:.3f} m)", flush=True)
                    elif _whStill > 0:
                        s.setdefault("hist", {})[arm] = []
                    if s["held"][arm] >= 20 and (float(a[g]) >= 0 or still):
                        ez = float((Rb.T @ (_w_t(robot.get_eef_position(arm=arm)) - bpos))[2])
                        if ez < _whMin:
                            # 04:50 fix: only block when the hand has CARRIED something up (risen >= 15 cm since the close);
                            # WALLHOLD run 1 blocked 3000 opens of a hand closed at table height on the detector's edge
                            # that never lifted it, freezing the retry for the rest of the episode
                            if s["carry"].get(arm, False) and s["blocked"] < _whMax and s["bh"].get(arm, 0) < 600:
                                s["bh"][arm] = s["bh"].get(arm, 0) + 1
                                a[g] = -1.0; s["blocked"] += 1
                                if s["blocked"] in (1, 10, 100) or s["blocked"] % 500 == 0:
                                    print(f"WALL-HOLD: {arm} open blocked at step {t} (eef {ez:.2f} m < {_whMin:.2f}; "
                                          f"#{s['blocked']})", flush=True)
                                continue
                        elif s["carry"].get(arm, False):
                            jq = _w_t(robot.get_joint_positions())
                            s["hist"] = {}
                            s.update(mode="lift", arm=arm, n=0, q=jq[_w_t(robot.arm_control_idx[arm]).long()].clone(),
                                     torso=jq[_w_t(robot.trunk_control_idx).long()].clone(), z0=ez, zl=ez)
                            print(f"WALL-HOLD: {arm} {'still' if still else 'open'} at the wall at step {t} (eef {ez:.3f} m) -> lifting to "
                                  f"{_whZ:.2f}", flush=True)
                            break
                    # 05:20: no upper width bound - WALLTWIST run 1 carried the detector by its wide bottom tier with the
                    # fingers reading +1.00 (fully open) the whole time, so "held" never counted; carry (>= 15 cm rise) guards
                    s["held"][arm] = s["held"][arm] + 1 if (float(a[g]) < 0 and w > -0.95) else 0
                    hz_now = float((Rb.T @ (_w_t(robot.get_eef_position(arm=arm)) - bpos))[2])
                    if s["held"][arm] == 1:
                        s["z_close"][arm] = hz_now; s["carry"][arm] = False; s["bh"][arm] = 0
                    elif s["held"][arm] == 0:
                        s["carry"][arm] = False
                    elif hz_now >= s["z_close"].get(arm, 9.0) + 0.15 and not s["carry"].get(arm, False):
                        s["carry"][arm] = True
                        print(f"WALL-HOLD: {arm} carrying (eef {s['z_close'][arm]:.2f} -> {hz_now:.2f} m) at step {t}",
                              flush=True)
            if s["mode"] in ("idle", "done"):
                return
            arm = s["arm"]; g = _WG[arm]
            a[0] = 0.0; a[1] = 0.0; a[2] = 0.0
            _w_set(a, slice(3, 7), s["torso"])
            sl = slice(7, 14) if arm == "left" else slice(15, 22)
            ep = _w_t(robot.get_eef_position(arm=arm))
            ez = float((Rb.T @ (ep - bpos))[2])
            s["n"] += 1
            m = s["mode"]
            if m == "release":
                a[g] = 1.0
                if s["n"] >= 60:
                    print(f"WALL-HOLD: released, model in charge at step {t}", flush=True)
                    s["mode"] = "idle"; s["held"] = {"left": 0, "right": 0}
                return
            a[g] = -1.0
            if m == "lift":
                if s["n"] % 40 == 0:
                    stall = ez - s["zl"] < 0.003; s["zl"] = ez
                else:
                    stall = False
                done_z = (abs(ez - _whZ) < 0.005) if _whExact else (ez >= _whZ)
                if _whExact and s["n"] % 40 == 0:
                    stall = abs(ez - s.get("zl2", 9.0)) < 0.003 and abs(ez - _whZ) >= 0.005
                    s["zl2"] = ez
                if done_z or stall or s["n"] >= 150:
                    print(f"WALL-HOLD: lift done at step {t} (eef {s['z0']:.3f} -> {ez:.3f} m"
                          f"{', stalled' if stall else ''})", flush=True)
                    if _wtOn and not s["twisted"]:
                        s["twisted"] = True
                        Re = _w_t(T.quat2mat(_w_t(robot.get_eef_orientation(arm=arm))))
                        col = int(th.argmin(th.abs(Re.T @ f)))          # eef axis most perpendicular to the wall normal
                        a0 = _w_t(robot.get_eef_position(arm=arm)).clone()
                        sv = th.stack([-f[1], f[0], th.zeros(())]); sv = sv / th.clamp(th.linalg.norm(sv), min=1e-6)
                        s.update(mode="twist", phase="back", k=0, n=0, f=f, col=col, r0=_w_roll(robot, arm, f, col),
                                 anchor=a0.clone(), a0=a0, sv=sv, pairs=list(_wtPairs))
                        if _wtDot:
                            try:
                                dn = _w_dot(self, arm)
                            except Exception:
                                dn = None
                            ds = ([x[1] for x in s.get("dots", [])[-10:]] if s.get("dot_arm", "left") == arm else []) + \
                                 ([dn[0]] if dn else [])
                            if ds:
                                # circular median-ish: mean of unit vectors
                                cx = sum(_wm.cos(_wm.radians(x)) for x in ds); cy = sum(_wm.sin(_wm.radians(x)) for x in ds)
                                dang = _wm.degrees(_wm.atan2(cy, cx)) % 360.0
                                need = (dang - _wtDotOff + 180.0) % 360.0 - 180.0
                                first = [(need + o, 0.0) for o in _wtDotOffs] + \
                                        [(need, dl) for dl in _wtDotLat]
                                s["pairs"] = first + [p for p in _wtPairs if all(abs(p[0] - q[0]) > 5 for q in first)]
                                print(f"DOT-ALIGN: red dot at {dang:.0f} deg in the {arm} wrist image ({len(ds)} reads, now "
                                      f"{'seen' if dn else 'not seen'}) -> turn {need:+.0f} deg first", flush=True)
                            else:
                                print(f"DOT-ALIGN: red dot not seen while carrying -> normal sweep", flush=True)
                        print(f"WALL-TWIST: sweep start at step {t} (roll0 {_wm.degrees(s['r0']):+.0f} deg, "
                              f"{len(s['pairs'])} tries, sideways offsets {_wtLat})", flush=True)
                    else:
                        s["mode"] = "release"; s["n"] = 0
                else:
                    dz = 0.001 if (not _whExact or ez < _whZ) else -0.001
                    s["q"] = _w_ik(robot, arm, s["q"], Rb, th.tensor([0.0, 0.0, dz]), zero)
            elif m == "twist":
                fw = s["f"]; ph = s["phase"]
                s["anchor"] = s["a0"] + s["pairs"][s["k"]][1] * s["sv"]
                # fast mode: (step, back/press/pull speeds, cycle lengths) + hold the hand on the anchor across the wall
                vb, nb, vp, np_, nst, nd, vpl, npl = ((0.002, 10, 0.002, 20, 5, 10, 0.0015, 10) if _wtFast
                                                      else (0.001, 20, 0.001, 35, 10, 15, 0.001, 15))
                hold = zero
                if _wtFast:
                    dv = s["anchor"] - ep
                    dv = dv - (dv @ fw) * fw
                    hold = th.clamp(0.5 * dv, -0.003, 0.003)

                if _wtFast and _wtReach > 0:
                    # v9 (11:35, WALLTWIST8b: during the turns the hand sat at 0.83-0.84 m from the base = full reach and
                    # drifted 22 cm along the wall): in EVERY sweep phase, base forward while the hand is beyond the reach
                    # radius, and base sideways (robot frame) toward the anchor when the hand is > 2 cm off it laterally
                    hd_b = float(th.linalg.norm((ep - bpos)[:2]))
                    if _wtPressCreep > 0 and ph == "press":
                        a[0] = max(float(a[0]), _wtPressCreep)
                    if hd_b > _wtReach and not (_wtCreepPress and ph not in ("turn", "press")):
                        a[0] = 0.08
                        s["creep"] = s.get("creep", 0) + 1
                        if s["creep"] in (1, 50) or s["creep"] % 200 == 0:
                            print(f"WALL-TWIST: base creep #{s['creep']} at step {t} (hand {hd_b:.2f} m from the base "
                                  f"> {_wtReach:.2f})", flush=True)
                    lat = Rb.T @ (s["anchor"] - ep)
                    if abs(float(lat[1])) > 0.02:
                        a[1] = max(-0.08, min(0.08, 1.0 * float(lat[1])))
                        s["side"] = s.get("side", 0) + 1
                        if s["side"] in (1, 50) or s["side"] % 200 == 0:
                            print(f"WALL-TWIST: base sidestep #{s['side']} at step {t} (hand {100 * float(lat[1]):+.1f} cm "
                                  f"off the anchor sideways)", flush=True)

                def _goto(off):
                    # v5 (07:50, WALLTWIST4: the hand drifted ~1 m back from the wall over 12 cycles): drive the hand to
                    # anchor + off along the wall normal (all three axes, P-control, <= 3 mm/step), so no cycle can drift
                    return th.clamp(0.5 * (s["anchor"] + off * fw - ep), -0.003, 0.003)
                if ph == "back":
                    if _wtFast:
                        s["q"] = _w_ik(robot, arm, s["q"], Rb, _goto(-_wtLatBack if s["pairs"][s["k"]][1] != 0.0 else -0.02), zero)
                    else:
                        s["q"] = _w_ik(robot, arm, s["q"], Rb, -vb * fw + hold, zero)
                    if s["n"] >= (nb if s["pairs"][s["k"]][1] == 0.0 else 30):
                        s["phase"] = "turn"; s["n"] = 0
                elif ph == "turn":
                    # unwrapped roll since the sweep start, so -180 -> +15 turns back through 0 (not on into the joint limit)
                    rn = _w_roll(robot, arm, fw, s["col"])
                    s["ru"] = s.get("ru", 0.0) + ((rn - s.get("rp", s["r0"]) + _wm.pi) % (2 * _wm.pi) - _wm.pi)
                    s["rp"] = rn
                    err = _wm.radians(s["pairs"][s["k"]][0]) - s["ru"]
                    rate = _wm.radians(_wtTurn if _wtFast else 1)
                    wv = max(-rate, min(rate, err))
                    s["q"] = _w_ik(robot, arm, s["q"], Rb, _goto(-0.02) if _wtFast else hold, wv * fw)
                    if _wtFast and s["n"] % 20 == 0:
                        stalled = abs(err) > abs(s.get("err20", 9.0)) - _wm.radians(2)
                        s["err20"] = err
                    else:
                        stalled = False
                    if abs(err) < _wm.radians(2) or s["n"] >= (90 if _wtFast else 200) or (stalled and s["n"] >= 20):
                        if abs(err) >= _wm.radians(2):
                            print(f"WALL-TWIST: turn to {s['pairs'][s['k']][0]:+.0f} stopped at {_wm.degrees(s.get('ru', 0.0)):+.0f} "
                                  f"(joint limit / stall) at step {t}", flush=True)
                        s["phase"] = "press"; s["n"] = 0; s["p0"] = ep.clone(); s["pl"] = ep.clone(); s["err20"] = 9.0
                elif ph == "press":
                    s["q"] = _w_ik(robot, arm, s["q"], Rb, (_goto(_wtPress) if _wtFast else vp * fw + hold), zero)
                    stop = False
                    if s["n"] % nst == 0:
                        # fast: contact only counts after 10 steps (the arm needs a few steps to get moving)
                        stop = float((ep - s["pl"]) @ fw) < (0.001 if _wtFast else 0.002) and (s["n"] >= 10 or not _wtFast)
                        s["pl"] = ep.clone()
                    reached = _wtFast and float((ep - s["anchor"]) @ fw) > _wtPress - 0.005
                    if stop or reached or s["n"] >= ((30 + int(_wtPress / 0.003)) if _wtFast else np_):
                        if _wtFast and _wtSeat > 0 and stop:
                            s["phase"] = "seat"; s["n"] = 0
                        else:
                            s["phase"] = "dwell"; s["n"] = 0
                elif ph == "seat":
                    # keep winding the command into the contact = press the detector onto what it touches
                    s["q"] = _w_ik(robot, arm, s["q"], Rb, 0.002 * fw + hold, zero)
                    if s["n"] >= _wtSeat:
                        s["phase"] = "dwell"; s["n"] = 0
                elif ph == "dwell":
                    s["q"] = _w_ik(robot, arm, s["q"], Rb, hold, zero)
                    if s["n"] >= nd:
                        s["phase"] = "pull"; s["n"] = 0; s["p0"] = ep.clone()
                        # 07:00 (WALLTWIST3 false lock at -30 deg): the press winds the joint command up to 3 cm into the
                        # wall; restart the pull from the MEASURED joints so the hand actually moves back
                        s["q"] = _w_t(robot.get_joint_positions())[_w_t(robot.arm_control_idx[arm]).long()].clone()
                elif ph == "pull":
                    s["q"] = _w_ik(robot, arm, s["q"], Rb, -vpl * fw + hold, zero)
                    pull_done = s["n"] >= npl
                    if _wtPullBase:
                        a[0] = -0.06
                        pull_done = s["n"] >= 40 or (s["n"] >= npl and float((s["p0"] - ep) @ fw) >= 0.015)
                    if pull_done:
                        moved = float((s["p0"] - ep) @ fw)
                        if _wtFast and float(th.linalg.norm(s["p0"] - ep)) >= 0.008:
                            moved = max(moved, 0.01)   # the hand clearly moved (any direction) -> not locked
                        roll = _wm.degrees(s.get("ru", 0.0))
                        print(f"WALL-TWIST: angle {s['pairs'][s['k']][0]:+.0f} side {100 * s['pairs'][s['k']][1]:+.0f} cm (roll {roll:+.0f}) tried at "
                              f"step {t}, pull-back moved {moved * 100:.1f} cm (eef z {ez:.3f})", flush=True)
                        if moved < 0.005 and _wtNoLock:
                            print(f"WALL-TWIST: pull blocked at step {t} (snag, not attached - an attachment would have "
                                  f"ended the episode) -> next angle", flush=True)
                            moved = 1.0
                        if moved < 0.005 and _wtFast and not s.get("confirm"):
                            # one more pull from the measured joints before believing it
                            print(f"WALL-TWIST: pull blocked at step {t} -> confirming with a second pull", flush=True)
                            s["confirm"] = True; s["n"] = 0; s["p0"] = ep.clone()
                            s["q"] = _w_t(robot.get_joint_positions())[_w_t(robot.arm_control_idx[arm]).long()].clone()
                        elif moved < 0.005:
                            print(f"WALL-TWIST: hand locked at step {t} -> detector on the nail? releasing", flush=True)
                            s["mode"] = "release"; s["n"] = 0
                        elif s["k"] + 1 >= len(s["pairs"]):
                            print(f"WALL-TWIST: all {len(s['pairs'])} tries done at step {t} -> releasing", flush=True)
                            s["mode"] = "release"; s["n"] = 0
                        else:
                            s["k"] += 1; s["phase"] = "back"; s["n"] = 0; s["confirm"] = False
            _w_set(a, sl, s["q"])

        eval_mod.Evaluator.step = _wh_step
        print(f"WALL-HOLD: rule on (hold below {_whMin:.2f} m, lift to {_whZ:.2f} m before release"
              f"{', WALL-TWIST sweep' if _wtOn else ''})", flush=True)

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
            # 2026-10-11 (user, task 35): B1K_SPAWN_YAW="inst:deg[,inst:deg]" turns a task object about the world vertical
            # axis, in place, right after the instance loads (e.g. camera_tripod.n.01_1:-25 so the camera's mount twist
            # lands inside AttachedTo's 15 deg window). Sim-only spawn change, logged.
            for _spec in [s for s in os.environ.get("B1K_SPAWN_YAW", "").split(",") if s]:
                _inst, _deg = _spec.rsplit(":", 1)
                if _inst in scope:
                    _p, _q = scope[_inst].get_position_orientation()
                    _dq = T.euler2quat(th.tensor([0.0, 0.0, float(_deg) * 3.14159265 / 180.0]))
                    scope[_inst].set_position_orientation(_p, T.quat_multiply(_dq, _q))
                    logger.info(f"SPAWN-YAW: {_inst} turned {float(_deg):+.1f} deg about z at {[round(float(v), 3) for v in _p]}")
                else:
                    logger.warning(f"SPAWN-YAW: {_inst} not in task scope {sorted(scope)}")

            # 2026-10-10: B1K_POSE_OBJ_EVERY (default 500) sets the object-record interval; object records now also carry
            # quaternions ("objq") and, for every scope pair that can attach (AttachedTo, e.g. camera -> tripod), the
            # smallest attachment-link gap ("att": {"child>parent": [pos_m, orn_deg]}); the sim attaches on contact when
            # pos < 0.05 m and orn < 15 deg.
            obj_every = int(os.environ.get("B1K_POSE_OBJ_EVERY", "500"))
            from omnigibson.object_states import AttachedTo as _AttachedTo

            def _att_gaps():
                out = {}
                for ka, a in scope.items():
                    if _AttachedTo not in getattr(a, "states", {}):
                        continue
                    for kb, b in scope.items():
                        if kb == ka or _AttachedTo not in getattr(b, "states", {}):
                            continue
                        try:
                            best = None
                            for cl, pls in a.states[_AttachedTo]._get_parent_candidates(b).items():
                                cp, co = a.states[_AttachedTo].links[cl].get_position_orientation()
                                for pl in pls:
                                    pp, po = b.states[_AttachedTo].links[pl].get_position_orientation()
                                    g = (float(th.norm(cp - pp)), float(T.get_orientation_diff_in_radian(co, po)) * 57.29578,
                                         [round(float(v), 3) for v in (pp - cp)])
                                    if best is None or g[0] < best[0]:
                                        best = g
                            if best is not None:
                                # 2026-10-13: + the world gap vector parent-link minus child-link (diagnostic only)
                                out[f"{ka}>{kb}"] = [round(best[0], 4), round(best[1], 1), best[2]]
                        except Exception:
                            pass
                return out

            def _log_pose(t, with_objs=False):
                pos, quat = evaluator.robot.get_position_orientation()
                yaw = float(T.quat2euler(quat)[2])
                rec = {"t": t, "base": [round(float(pos[0]), 3), round(float(pos[1]), 3), round(yaw, 4)]}
                # 2026-10-11: world positions of both end effectors (grasp-miss diagnosis: too high vs off to the side)
                try:
                    rec["eef"] = {arm: [round(float(v), 3) for v in evaluator.robot.get_eef_position(arm=arm)]
                                  for arm in ("left", "right")}
                except Exception:
                    pass
                if with_objs or t % obj_every == 0:
                    objs, objq = {}, {}
                    for k, v in scope.items():
                        try:
                            op, oq = v.get_position_orientation()
                            objs[k] = [round(float(op[0]), 3), round(float(op[1]), 3), round(float(op[2]), 3)]
                            objq[k] = [round(float(q), 4) for q in oq]
                        except Exception:
                            pass
                    rec["objs"] = objs
                    rec["objq"] = objq
                    att = _att_gaps()
                    if att:
                        rec["att"] = att
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
