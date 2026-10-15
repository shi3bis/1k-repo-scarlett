"""Run scripts/serve_b1k.py with correlated flow-matching noise OFF (B1K_INDEP_NOISE=1, default on).

The task checkpoints were fine-tuned with use_correlated_noise=False, but pi_behavior_b1k_fast_100task
inherits use_correlated_noise=True; serving with it on gives wild arm outputs (notepad 2026-09-19).
Repo is left untouched: we patch the config registry, then hand off to serve_b1k's main.
"""
import dataclasses
import os
import runpy
import sys

REPO = "/workspace/persistent/repos/place_inside_4_tasks"
sys.path.insert(0, os.path.join(REPO, "scripts"))

from b1k.training import config as _config  # noqa: E402

if os.environ.get("B1K_INDEP_NOISE", "1") == "1":
    for name, cfg in list(_config._CONFIGS_DICT.items()):
        _config._CONFIGS_DICT[name] = dataclasses.replace(
            cfg, model=dataclasses.replace(cfg.model, use_correlated_noise=False)
        )
    print("serve_indep: use_correlated_noise=False for all configs", flush=True)

if os.environ.get("B1K_NO_INPAINT", "0") == "1":
    # Rolling inpainting off: the wrapper's next_initial_actions always reads as None (writes ignored).
    from b1k.shared import eval_b1k_wrapper as _w
    _w.B1KPolicyWrapper.next_initial_actions = property(lambda self: None, lambda self, v: None)
    print("serve_indep: rolling inpainting disabled", flush=True)

if os.environ.get("B1K_PIN_STAGE", "") != "":
    # Stage pinned: the wrapper's current_stage always reads B1K_PIN_STAGE (writes from voting/correction rules ignored).
    from b1k.shared import eval_b1k_wrapper as _w
    _pin = int(os.environ["B1K_PIN_STAGE"])
    _w.B1KPolicyWrapper.current_stage = property(lambda self: _pin, lambda self, v: None)
    print(f"serve_indep: stage pinned to {_pin}", flush=True)

if os.environ.get("B1K_BASE50", "0") == "1":
    # Original 2025 50-task checkpoint (IliaLarchenko/behavior_50t_checkpoint): its task_stage_embeddings table has
    # 596 rows (tasks 0-49). Today's TASK_NUM_STAGES was extended to 627 rows for tasks 55/76/82, so restore the
    # first-50-task slice in the model module's bound copies before the model is built (offsets for 0-49 unchanged).
    from b1k.models import pi_behavior as _pb, pi_behavior_config as _pbc
    _first50 = tuple(_pbc.TASK_NUM_STAGES[:50])
    _pb.TASK_NUM_STAGES = _first50
    _pb.TOTAL_TASK_STAGE_EMBEDDINGS = sum(_first50)
    _pb.TASK_STAGE_OFFSETS = tuple([0] + [sum(_first50[: i + 1]) for i in range(len(_first50) - 1)])
    print(f"serve_indep: base-50 stage table ({_pb.TOTAL_TASK_STAGE_EMBEDDINGS} rows)", flush=True)

if os.environ.get("B1K_CHAIN_76_80_87", "0") == "1":
    # shi2bis/tasks_76_80_87 (chain-76-80-87 branch): task_stage_embeddings has 649 rows = today's 627 + 80=10 + 87=12
    # (README; verified from the ckpt metadata 2026-10-03). Rows for tasks <= 79 keep their offsets.
    from b1k.models import pi_behavior as _pb, pi_behavior_config as _pbc
    _t = list(_pbc.TASK_NUM_STAGES); _t[80] = 10; _t[87] = 12; _t = tuple(_t)
    _pbc.TASK_NUM_STAGES = _pb.TASK_NUM_STAGES = _t
    _pb.TOTAL_TASK_STAGE_EMBEDDINGS = sum(_t)
    _pb.TASK_STAGE_OFFSETS = tuple([0] + [sum(_t[: i + 1]) for i in range(len(_t) - 1)])
    print(f"serve_indep: chain-76-80-87 stage table ({_pb.TOTAL_TASK_STAGE_EMBEDDINGS} rows)", flush=True)

if os.environ.get("B1K_CHAIN_35_63_78", "0") == "1":
    # 2026-10-09: shi2bis/joint_35_63_78 (branch attach-35-63-78): 656 rows = 649-row 76-80-87 table + 63=3 + 78=4
    # (READY "serve_with"; ext100 base gives 63/78 zero stages). Use INSTEAD of B1K_CHAIN_76_80_87.
    from b1k.models import pi_behavior as _pb, pi_behavior_config as _pbc
    _t = list(_pbc.TASK_NUM_STAGES); _t[80] = 10; _t[87] = 12; _t[63] = 3; _t[78] = 4; _t = tuple(_t)
    _pbc.TASK_NUM_STAGES = _pb.TASK_NUM_STAGES = _t
    _pb.TOTAL_TASK_STAGE_EMBEDDINGS = sum(_t)
    _pb.TASK_STAGE_OFFSETS = tuple([0] + [sum(_t[: i + 1]) for i in range(len(_t) - 1)])
    print(f"serve_indep: attach-35-63-78 stage table ({_pb.TOTAL_TASK_STAGE_EMBEDDINGS} rows)", flush=True)

if os.environ.get("B1K_COV4_57_60_69_93", "0") == "1":
    # 2026-10-10: shi2bis/joint_57_60_69_93 (branch covered-57-60-69-93): 673 rows = 656-row attach table + 57=5, 60=4,
    # 69=3, 93=5 (READY "serve_with"). Use INSTEAD of B1K_CHAIN_76_80_87 / B1K_CHAIN_35_63_78.
    from b1k.models import pi_behavior as _pb, pi_behavior_config as _pbc
    _t = list(_pbc.TASK_NUM_STAGES); _t[80] = 10; _t[87] = 12; _t[63] = 3; _t[78] = 4
    _t[57] = 5; _t[60] = 4; _t[69] = 3; _t[93] = 5; _t = tuple(_t)
    _pbc.TASK_NUM_STAGES = _pb.TASK_NUM_STAGES = _t
    _pb.TOTAL_TASK_STAGE_EMBEDDINGS = sum(_t)
    _pb.TASK_STAGE_OFFSETS = tuple([0] + [sum(_t[: i + 1]) for i in range(len(_t) - 1)])
    print(f"serve_indep: covered-57-60-69-93 stage table ({_pb.TOTAL_TASK_STAGE_EMBEDDINGS} rows)", flush=True)

if os.environ.get("B1K_SEED", "") != "":
    # Inference RNG seed: openpi's Policy defaults to jax.random.key(0) on every server start, so repeated evals of the
    # same instance replay the same noise. Seed the policy RNG from B1K_SEED instead.
    import jax
    from openpi.policies import policy as _op
    _seed = int(os.environ["B1K_SEED"])
    _orig_init = _op.Policy.__init__
    def _seeded_init(self, *a, **kw):
        _orig_init(self, *a, **kw)
        self._rng = jax.random.key(_seed)
    _op.Policy.__init__ = _seeded_init
    print(f"serve_indep: inference seed {_seed}", flush=True)

if os.environ.get("B1K_RELEASE_STAGE", "") != "":
    # Eval-time release rule (added 2026-10-04, task 76 dispose_of_glass): the model carries the trash can to the drop
    # spot but does not let go (offline probe: release keyed on proprio, not vision). When the stage tracker is at
    # >= B1K_RELEASE_STAGE and BOTH grippers are holding something (measured width strictly between closed and open,
    # = the two-handed bin grasp; the glass is one-handed so it is never dropped), force both gripper commands open
    # for the whole chunk. Arms/base keep the model's prediction. Re-checked every chunk from the measured state.
    import logging as _lg
    import numpy as _np2
    from b1k.shared import eval_b1k_wrapper as _rw
    _rs = int(os.environ["B1K_RELEASE_STAGE"]); _rtask = int(os.environ.get("B1K_RELEASE_TASK", "76"))
    _hold = lambda g: -0.9 < g < 0.9
    _orig_rules = _rw.apply_correction_rules
    def _release_rules(task_id, stage, state, actions):
        actions, stage = _orig_rules(task_id, stage, state, actions)
        if task_id == _rtask and stage >= _rs and _hold(state[14]) and _hold(state[22]):
            actions = _np2.array(actions, copy=True)
            actions[:, 14] = 1.0; actions[:, 22] = 1.0
            _lg.getLogger("b1k").info(f"RELEASE RULE: task {task_id} stage {stage} grippers L {state[14]:+.2f} R {state[22]:+.2f} -> forced open")
        return actions, stage
    _rw.apply_correction_rules = _release_rules
    print(f"serve_indep: release rule on (task {_rtask}, stage >= {_rs}, both grippers holding)", flush=True)

if os.environ.get("B1K_EMPTY_OPEN_N", "") != "":
    # Empty-gripper re-open rule (added 2026-10-04). "Holding" is read from the measured finger width in the proprio the
    # server already receives (normalized -1 closed .. +1 open): fingers that close on an object stop at the object's
    # width; fingers that close on nothing go fully shut. So width < B1K_EMPTY_THR (default -0.9) = closed and EMPTY.
    # When a hand has been closed-and-empty for B1K_EMPTY_OPEN_N consecutive steps (0 = immediately), its gripper command
    # is forced to +1 (open) every step until the measured width is back above +0.9, then control returns to the model.
    # Also opens that hand in the rolling-inpainting carry-over so the next chunk does not start closed. Per hand.
    import logging as _lg2
    import numpy as _np3
    from b1k.shared import eval_b1k_wrapper as _ew
    from b1k.policies.b1k_policy import extract_state_from_proprio as _esp2
    _eN = int(os.environ["B1K_EMPTY_OPEN_N"]); _eT = float(os.environ.get("B1K_EMPTY_THR", "-0.9"))
    # 2026-10-10 "nudged retry" (optional): every time a hand is forced open after an empty close, add a small sideways
    # base velocity (robot-frame y, alternating left/right per retry) for B1K_EMPTY_NUDGE_STEPS steps at B1K_EMPTY_NUDGE_V m/s
    # (defaults 20 steps x 0.06 m/s ~ 4 cm) on top of the model's command, so the next attempt is not an exact repeat.
    _nS = int(os.environ.get("B1K_EMPTY_NUDGE_STEPS", "0")); _nV = float(os.environ.get("B1K_EMPTY_NUDGE_V", "0.06"))
    # 2026-10-15 B1K_EMPTY_HANDS (default "LR"): which hands the rule watches. Task 0 uses "R": the L hand presses the radio
    # button with CLOSED (= empty-reading) fingers, which the rule must not open.
    _eHands = os.environ.get("B1K_EMPTY_HANDS", "LR")
    _elog = _lg2.getLogger("b1k")
    _orig_act_e, _orig_reset_e = _ew.B1KPolicyWrapper.act, _ew.B1KPolicyWrapper.reset
    def _e_reset(self, *a, **kw):
        self._eo = {14: [0, False], 22: [0, False]}
        self._nudge = [0, 1.0]
        return _orig_reset_e(self, *a, **kw)
    def _e_act(self, obs):
        if not hasattr(self, "_eo"):
            self._eo = {14: [0, False], 22: [0, False]}
        if not hasattr(self, "_nudge"):
            self._nudge = [0, 1.0]
        st = _np3.asarray(_esp2(_np3.asarray(obs["robot_r1::proprio"]))).reshape(-1)
        out = _orig_act_e(self, obs)
        for g, name in ((14, "L"), (22, "R")):
            if name not in _eHands:
                continue
            c = self._eo[g]
            if c[1]:
                if st[g] > 0.9:
                    c[0], c[1] = 0, False
                    _elog.info(f"EMPTY-OPEN: {name} open again (w {st[g]:+.2f}) at step {self.step_count} -> model control")
            else:
                c[0] = c[0] + 1 if st[g] < _eT else 0
                if st[g] < _eT and c[0] > _eN:
                    c[1] = True
                    _elog.info(f"EMPTY-OPEN: {name} closed+empty (w {st[g]:+.2f}) {c[0]} steps at step {self.step_count} -> forcing open")
                    if _nS > 0:
                        self._nudge = [_nS, -self._nudge[1]]          # alternate left / right per retry
                        _elog.info(f"NUDGE: {'left' if self._nudge[1] > 0 else 'right'} {_nV:.2f} m/s for {_nS} steps")
            if c[1]:
                out[g] = 1.0
                if self.next_initial_actions is not None:
                    self.next_initial_actions[:, g] = 1.0
        if self._nudge[0] > 0:
            out[1] = float(out[1]) + self._nudge[1] * _nV
            self._nudge[0] -= 1
        return out
    _ew.B1KPolicyWrapper.act, _ew.B1KPolicyWrapper.reset = _e_act, _e_reset
    print(f"serve_indep: empty-gripper re-open rule on (N={_eN} steps, empty if width < {_eT})"
          + (f", nudged retry {_nS} steps x {_nV} m/s" if _nS > 0 else ""), flush=True)

if os.environ.get("B1K_RETRY_WINDOW", "") != "":
    # Back-off-and-retry rule (added 2026-10-09, aimed at task 0: the robot reaches the radio fast, then parks 0.5-0.8 m
    # away without a lasting grasp). The server does not know where the object is, so the trigger is behavioural:
    #   the base has moved < B1K_RETRY_MOVE m over the last B1K_RETRY_WINDOW steps AND no hand has been holding something
    #   (width between closed and open while commanded closed) for >= 30 consecutive steps in that window.
    # Then, for B1K_RETRY_BACK steps, the base command is replaced by a straight backward drive at B1K_RETRY_V m/s (robot
    # frame; arms/grippers keep the model's output), after which control returns to the model so it re-approaches from a
    # new, hopefully more demo-like pose. Cooldown = one full window; at most B1K_RETRY_MAX back-offs per episode.
    import collections as _col
    import logging as _lg3
    import numpy as _np4
    from b1k.shared import eval_b1k_wrapper as _rw2
    from b1k.policies.b1k_policy import extract_state_from_proprio as _esp3
    _rW = int(os.environ["B1K_RETRY_WINDOW"]); _rM = float(os.environ.get("B1K_RETRY_MOVE", "0.15"))
    _rB = int(os.environ.get("B1K_RETRY_BACK", "45")); _rV = float(os.environ.get("B1K_RETRY_V", "0.2"))
    _rMax = int(os.environ.get("B1K_RETRY_MAX", "6"))
    _rlog = _lg3.getLogger("b1k")
    _orig_act_r, _orig_reset_r = _rw2.B1KPolicyWrapper.act, _rw2.B1KPolicyWrapper.reset

    def _r_init(self):
        self._rt = {"xy": _col.deque(maxlen=_rW), "hold": 0, "held_long": 0, "back": 0, "cool": 0, "n": 0}

    def _r_reset(self, *a, **kw):
        _r_init(self)
        return _orig_reset_r(self, *a, **kw)

    def _r_act(self, obs):
        if not hasattr(self, "_rt"):
            _r_init(self)
        rt = self._rt
        prop = _np4.asarray(obs["robot_r1::proprio"]).reshape(-1)
        st = _np4.asarray(_esp3(prop)).reshape(-1)
        out = _orig_act_r(self, obs)
        if prop.shape[0] >= 247:                      # 256-d eval layout: base_qpos = [244:247] (x, y, yaw)
            rt["xy"].append(prop[244:246].copy())
        holding = any(-0.9 < st[g] < 0.9 and float(out[g]) < 0 for g in (14, 22))
        rt["hold"] = rt["hold"] + 1 if holding else 0
        if rt["hold"] >= 30:
            rt["held_long"] = len(rt["xy"])             # a real grasp happened: restart the window
        if rt["back"] > 0:                               # backing off: override the base command
            out[0], out[1], out[2] = -_rV, 0.0, 0.0
            rt["back"] -= 1
            if rt["back"] == 0:
                rt["xy"].clear(); rt["held_long"] = 0; rt["cool"] = _rW
                _rlog.info(f"RETRY: back-off done at step {self.step_count}, control back to the model")
            return out
        if rt["cool"] > 0:
            rt["cool"] -= 1
            return out
        if rt["held_long"] > 0:
            rt["held_long"] -= 1
            return out
        if rt["n"] < _rMax and len(rt["xy"]) == _rW:
            pts = _np4.stack(rt["xy"])
            moved = float(_np4.linalg.norm(pts.max(0) - pts.min(0)))
            if moved < _rM:
                rt["back"] = _rB; rt["n"] += 1
                _rlog.info(f"RETRY: parked {_rW} steps (moved {moved:.2f} m) without a grasp at step {self.step_count}"
                           f" -> backing off {_rB} steps at {_rV} m/s (retry {rt['n']}/{_rMax})")
        return out

    _rw2.B1KPolicyWrapper.act, _rw2.B1KPolicyWrapper.reset = _r_act, _r_reset
    print(f"serve_indep: retry rule on (window {_rW}, move < {_rM} m, back {_rB} steps at {_rV} m/s, max {_rMax})", flush=True)

if os.environ.get("B1K_FREEZE_FREE_ARM_TASK", "") != "":
    # Freeze-the-free-arm rule (added 2026-10-09, user request for task 35 attach_a_camera_to_a_tripod: the free (right) arm
    # rose while the other hand held the camera and lifted the tripod, so the camera could not land evenly on the stand).
    # For task B1K_FREEZE_FREE_ARM_TASK: once one hand has held something (width strictly between closed and open while
    # commanded closed) for >= B1K_FREEZE_HOLD_STEPS consecutive steps, the OTHER arm (7 joints) and its gripper are pinned
    # to their measured positions at that moment, every step (arm actions are absolute joint positions; Ilia's rules hold
    # still the same way, actions = state). Released when the holding hand opens again (width > 0.9), i.e. after the
    # attach/drop. The holding arm, torso and base keep the model's output. The inpainting carry-over is pinned too.
    import logging as _lg4
    import numpy as _np5
    from b1k.shared import eval_b1k_wrapper as _fw
    from b1k.policies.b1k_policy import extract_state_from_proprio as _esp4
    _fTask = int(os.environ["B1K_FREEZE_FREE_ARM_TASK"]); _fHold = int(os.environ.get("B1K_FREEZE_HOLD_STEPS", "30"))
    _ARM = {14: slice(7, 14), 22: slice(15, 22)}           # gripper index -> its arm joints
    _flog = _lg4.getLogger("b1k")
    _orig_act_f, _orig_reset_f = _fw.B1KPolicyWrapper.act, _fw.B1KPolicyWrapper.reset

    def _f_init(self):
        self._fz = {"hold": {14: 0, 22: 0}, "holder": None, "frozen": None, "pose": None}

    def _f_reset(self, *a, **kw):
        _f_init(self)
        return _orig_reset_f(self, *a, **kw)

    def _f_act(self, obs):
        if not hasattr(self, "_fz"):
            _f_init(self)
        fz = self._fz
        st = _np5.asarray(_esp4(_np5.asarray(obs["robot_r1::proprio"]))).reshape(-1)
        out = _orig_act_f(self, obs)
        if self.task_id != _fTask:
            return out
        for g in (14, 22):
            fz["hold"][g] = fz["hold"][g] + 1 if (-0.9 < st[g] < 0.9 and float(out[g]) < 0) else 0
        if fz["frozen"] is None:
            for g in (14, 22):
                other = 22 if g == 14 else 14
                if fz["hold"][g] >= _fHold and fz["hold"][other] < _fHold:
                    fz["holder"], fz["frozen"] = g, other
                    fz["pose"] = _np5.concatenate([st[_ARM[other]], [st[other]]]).astype(_np5.float32)
                    _flog.info(f"FREEZE: {'L' if g == 14 else 'R'} hand holding -> pinning the "
                               f"{'R' if other == 22 else 'L'} arm at step {self.step_count}")
                    break
        elif st[fz["holder"]] > 0.9:                       # holder opened: attach done / object dropped
            _flog.info(f"FREEZE: released at step {self.step_count} (holding hand opened)")
            fz["frozen"] = None
        if fz["frozen"] is not None:
            a, g = _ARM[fz["frozen"]], fz["frozen"]
            out[a] = __import__("torch").as_tensor(fz["pose"][:7], dtype=out.dtype)
            out[g] = float(fz["pose"][7])
            if self.next_initial_actions is not None:
                self.next_initial_actions[:, a] = fz["pose"][:7]
                self.next_initial_actions[:, g] = fz["pose"][7]
        return out

    _fw.B1KPolicyWrapper.act, _fw.B1KPolicyWrapper.reset = _f_act, _f_reset
    print(f"serve_indep: freeze-free-arm rule on (task {_fTask}, after {_fHold} holding steps)", flush=True)

if os.environ.get("B1K_ROOM_STAGE_TASK", "") != "":
    # Room-entry stage override (added 2026-10-10, user request for task 76: when the robot enters the bathroom, switch the
    # stage straight to the "set the bin down" stage). The server reads the WORLD base x, y from obs key
    # "robot_r1::world_base_pose" (sent by scripts/run_custom_eval.py) and looks the room up in the scene's floor_semseg_0.png (0.01 m/px, origin at the image centre, row = y).
    # On every entry into room B1K_ROOM_STAGE_ROOM (optionally only while a hand is holding something,
    # B1K_ROOM_STAGE_HOLDING=1), the wrapper's current_stage is set to B1K_ROOM_STAGE if it is lower, and the stage-vote
    # history is cleared, so the next model call gets that stage. The stage tracker continues normally afterwards.
    import logging as _lg5
    import numpy as _np6
    import matplotlib.image as _mimg
    from b1k.shared import eval_b1k_wrapper as _sw
    from b1k.policies.b1k_policy import extract_state_from_proprio as _esp5
    _sTask = int(os.environ["B1K_ROOM_STAGE_TASK"]); _sRoom = os.environ.get("B1K_ROOM_STAGE_ROOM", "bathroom")
    _sStage = int(os.environ.get("B1K_ROOM_STAGE", "-1")); _sHold = os.environ.get("B1K_ROOM_STAGE_HOLDING", "1") == "1"
    _sRel = int(os.environ.get("B1K_ROOM_RELEASE", "0"))   # >0: also force both grippers open for this many steps
    _A = "/workspace/persistent/BEHAVIOR-1K-v3.7/datasets/behavior-1k-assets"
    _scene = os.environ.get("B1K_ROOM_STAGE_SCENE", "hotel_suite_large")
    _cats = [l.strip() for l in open(f"{_A}/metadata/room_categories.txt")]
    _sem = (_mimg.imread(f"{_A}/scenes/{_scene}/layout/floor_semseg_0.png") * 255).round().astype(int)
    if _sem.ndim == 3:
        _sem = _sem[..., 0]
    _H = _sem.shape[0]
    _sid = _cats.index(_sRoom) + 1
    _slog = _lg5.getLogger("b1k")

    def _room_is_target(x, y):
        r, c = int(round(y / 0.01 + _H / 2)), int(round(x / 0.01 + _H / 2))
        if not (0 <= r < _H and 0 <= c < _H):
            return False
        w = _sem[max(0, r - 30):r + 30, max(0, c - 30):c + 30]       # nearest labelled pixel within 0.3 m (doorways)
        v = _sem[r, c]
        if v == 0:
            nz = _np6.argwhere(w > 0)
            if len(nz) == 0:
                return False
            d = ((nz - [min(r, 30), min(c, 30)]) ** 2).sum(1)
            v = w[tuple(nz[d.argmin()])]
        return v == _sid

    _orig_act_s, _orig_reset_s = _sw.B1KPolicyWrapper.act, _sw.B1KPolicyWrapper.reset

    def _s_reset(self, *a, **kw):
        self._rs = {"inside": False, "n": 0, "rel_left": 0, "rel_done": 0}
        return _orig_reset_s(self, *a, **kw)

    def _s_act(self, obs):
        if not hasattr(self, "_rs"):
            self._rs = {"inside": False, "n": 0, "rel_left": 0, "rel_done": 0}
        prop = _np6.asarray(obs["robot_r1::proprio"]).reshape(-1)
        wp = obs.get("robot_r1::world_base_pose") if hasattr(obs, "get") else None
        # 2026-10-10 fix: proprio base_qpos is measured from the robot's load pose, not world coordinates -> use the world
        # pose that run_custom_eval.py now sends; without it the rule is disabled (no wrong-frame room lookups)
        if self.task_id == _sTask and wp is not None:
            wp = _np6.asarray(wp).reshape(-1)
            inside = bool(_room_is_target(float(wp[0]), float(wp[1])))
            st = _np6.asarray(_esp5(prop)).reshape(-1)
            holding = any(-0.9 < st[g] < 0.9 for g in (14, 22))
            if (_sRel > 0 and inside and not self._rs["inside"] and holding and self._rs["rel_done"] == 0):
                # release-on-entry variant: force both grippers open for B1K_ROOM_RELEASE steps (first entry only); the
                # offline probe shows the model finishes a release 5/5 once a gripper already reads open
                self._rs["rel_left"] = _sRel; self._rs["rel_done"] = 1
                _slog.info(f"ROOM-RELEASE: entered {_sRoom} holding at step {self.step_count} -> forcing both grippers open"
                           f" for {_sRel} steps")
            if inside and not self._rs["inside"] and (holding or not _sHold) and self.current_stage < _sStage:
                _slog.info(f"ROOM-STAGE: entered {_sRoom} at step {self.step_count} (holding={holding}) -> stage "
                           f"{self.current_stage} -> {_sStage}")
                self.current_stage = _sStage
                self.prediction_history.clear()
                self._rs["n"] += 1
            self._rs["inside"] = inside
        out = _orig_act_s(self, obs)
        if self._rs.get("rel_left", 0) > 0:
            out[14] = 1.0; out[22] = 1.0; self._rs["rel_left"] -= 1
            if self.next_initial_actions is not None:
                self.next_initial_actions[:, 14] = 1.0; self.next_initial_actions[:, 22] = 1.0
        return out

    _sw.B1KPolicyWrapper.act, _sw.B1KPolicyWrapper.reset = _s_act, _s_reset
    print(f"serve_indep: room-stage rule on (task {_sTask}: entering {_sRoom}"
          f"{' while holding' if _sHold else ''} -> stage {_sStage}; scene {_scene})", flush=True)

if os.environ.get("B1K_PIN_TRIPOD_TASK", "") != "":
    # Pin-the-tripod-hand rule (added 2026-10-10, user request for task 35): once the B1K_PIN_HAND hand (default R, the
    # demo's tripod hand) has been HOLDING something (measured width strictly between -0.9 and +0.9 while commanded closed)
    # for >= B1K_PIN_HOLD_STEPS consecutive steps, its 7 arm joints are pinned at their measured positions (absolute joint
    # actions) and its gripper command is kept closed (-1), so the tripod is held still on the ground while the other hand
    # brings the camera. Released when the OTHER hand opens (> 0.9) after having held something (attach done / camera
    # dropped), or after B1K_PIN_MAX steps. Torso, base and the other arm keep the model's output. Carry-over pinned too.
    import logging as _lg6
    import numpy as _np7
    import torch as _th7
    from b1k.shared import eval_b1k_wrapper as _pw
    from b1k.policies.b1k_policy import extract_state_from_proprio as _esp6
    _pTask = int(os.environ["B1K_PIN_TRIPOD_TASK"]); _pHand = os.environ.get("B1K_PIN_HAND", "R").upper()
    _pN = int(os.environ.get("B1K_PIN_HOLD_STEPS", "10")); _pMax = int(os.environ.get("B1K_PIN_MAX", "3000"))
    _pg, _parm = (22, slice(15, 22)) if _pHand == "R" else (14, slice(7, 14))
    _og = 14 if _pg == 22 else 22
    _plog = _lg6.getLogger("b1k")
    _orig_act_p, _orig_reset_p = _pw.B1KPolicyWrapper.act, _pw.B1KPolicyWrapper.reset

    def _p_init(self):
        self._pt = {"hold": 0, "pose": None, "left": 0, "other_held": False}

    def _p_reset(self, *a, **kw):
        _p_init(self)
        return _orig_reset_p(self, *a, **kw)

    def _p_act(self, obs):
        if not hasattr(self, "_pt"):
            _p_init(self)
        pt = self._pt
        st = _np7.asarray(_esp6(_np7.asarray(obs["robot_r1::proprio"]))).reshape(-1)
        out = _orig_act_p(self, obs)
        if self.task_id != _pTask:
            return out
        if -0.9 < st[_og] < 0.9:
            pt["other_held"] = True
        if pt["pose"] is None:
            pt["hold"] = pt["hold"] + 1 if (-0.9 < st[_pg] < 0.9 and float(out[_pg]) < 0) else 0
            if pt["hold"] >= _pN:
                pt["pose"] = st[_parm].astype(_np7.float32).copy(); pt["left"] = _pMax
                _plog.info(f"PIN-TRIPOD: {_pHand} hand holding (w {st[_pg]:+.2f}) {pt['hold']} steps at step {self.step_count}"
                           f" -> pinning the {_pHand} arm, gripper kept closed")
        else:
            pt["left"] -= 1
            if (pt["other_held"] and st[_og] > 0.9) or pt["left"] <= 0:
                _plog.info(f"PIN-TRIPOD: released at step {self.step_count} "
                           f"({'other hand opened' if pt['left'] > 0 else 'max duration'})")
                pt["pose"] = None; pt["hold"] = 0; pt["other_held"] = False
        if pt["pose"] is not None:
            out[_parm] = _th7.as_tensor(pt["pose"], dtype=out.dtype)
            out[_pg] = -1.0
            if self.next_initial_actions is not None:
                self.next_initial_actions[:, _parm] = pt["pose"]; self.next_initial_actions[:, _pg] = -1.0
        return out

    _pw.B1KPolicyWrapper.act, _pw.B1KPolicyWrapper.reset = _p_act, _p_reset
    print(f"serve_indep: pin-tripod-hand rule on (task {_pTask}, {_pHand} hand, after {_pN} holding steps, max {_pMax})",
          flush=True)

if os.environ.get("B1K_HOLD_L_UNTIL_R_TASK", "") != "":
    # Hold-the-camera-arm rule (added 2026-10-10, user request for task 35): the L hand picks up the camera first (demos
    # and every eval), then the R hand must grip the tripod before the camera goes onto it. Evals instead swing the camera
    # over / into the tripod before R grips. So: once L has been HOLDING (width in (-0.9, 0.9)) for B1K_HOLD_L_STEPS steps,
    # the L arm (7:14) is pinned at its measured joints and the L gripper kept closed, until R has been holding for
    # B1K_HOLD_L_STEPS steps (tripod gripped) -> L back to the model for good. Also released if L loses the camera
    # (width < -0.9) or after B1K_HOLD_L_MAX steps. Base, torso and the R arm keep the model's output.
    import logging as _lg9
    import numpy as _np10
    import torch as _th10
    from b1k.shared import eval_b1k_wrapper as _hw
    from b1k.policies.b1k_policy import extract_state_from_proprio as _esp10
    _hTask = int(os.environ["B1K_HOLD_L_UNTIL_R_TASK"]); _hN = int(os.environ.get("B1K_HOLD_L_STEPS", "10"))
    _hMax = int(os.environ.get("B1K_HOLD_L_MAX", "4000"))
    # 2026-10-10 (user: "send the left arm down so it doesn't knock stuff over"): optional target pose for the frozen L arm
    # (7 absolute joints) reached by linear interpolation over B1K_HOLD_L_RAMP steps from the measured pose. Default target =
    # a real task-35 demo frame of the low, tucked camera carry (episode 7015, "move to tripod": L wrist ~0.31 m forward,
    # 0.31 m left, 0.67 m up in the robot frame), picked from carry frames with wrist z in the lowest 30% and x below median.
    _hPoseS = os.environ.get("B1K_HOLD_L_POSE", "")
    _hPose = (None if _hPoseS == "" else
              _np10.array([-0.204, 0.060, -0.115, -0.001, 0.495, 0.013, -0.117] if _hPoseS == "demo_low"
                          else [float(x) for x in _hPoseS.split(",")], dtype=_np10.float32))
    _hRamp = max(1, int(os.environ.get("B1K_HOLD_L_RAMP", "90")))
    _hlog = _lg9.getLogger("b1k")
    _orig_act_h, _orig_reset_h = _hw.B1KPolicyWrapper.act, _hw.B1KPolicyWrapper.reset

    def _h_init(self):
        self._hl = {"l": 0, "r": 0, "pose": None, "left": 0, "done": False, "start": None, "k": 0}

    def _h_reset(self, *a, **kw):
        _h_init(self)
        return _orig_reset_h(self, *a, **kw)

    def _h_act(self, obs):
        if not hasattr(self, "_hl"):
            _h_init(self)
        h = self._hl
        st = _np10.asarray(_esp10(_np10.asarray(obs["robot_r1::proprio"]))).reshape(-1)
        out = _orig_act_h(self, obs)
        if self.task_id != _hTask or h["done"]:
            return out
        # holding = width strictly inside (-0.9, 0.9) WHILE commanded closed (2026-10-11 fix: width alone also counted a
        # half-open hovering gripper, e.g. +0.88, as holding -> spurious freezes)
        h["l"] = h["l"] + 1 if (-0.9 < st[14] < 0.9 and float(out[14]) < 0) else 0
        h["r"] = h["r"] + 1 if (-0.9 < st[22] < 0.9 and float(out[22]) < 0) else 0
        if h["r"] >= _hN:
            if h["pose"] is not None:
                _hlog.info(f"HOLD-L: R holding {h['r']} steps (tripod) at step {self.step_count} -> L arm back to the model")
            h["pose"] = None; h["done"] = True
            return out
        if h["pose"] is None:
            if h["l"] >= _hN:
                h["pose"] = st[7:14].astype(_np10.float32).copy(); h["left"] = _hMax
                h["start"] = h["pose"].copy(); h["k"] = 0
                _hlog.info(f"HOLD-L: L holding (w {st[14]:+.2f}) at step {self.step_count}, R not yet -> freezing the L arm")
        else:
            h["left"] -= 1
            if st[14] < -0.9 or h["left"] <= 0:
                _hlog.info(f"HOLD-L: released at step {self.step_count} "
                           f"({'L lost the object' if st[14] < -0.9 else 'max duration'})")
                h["pose"] = None; h["l"] = 0
        if h["pose"] is not None and _hPose is not None:
            h["k"] += 1
            h["pose"] = h["start"] + (_hPose - h["start"]) * min(1.0, h["k"] / _hRamp)
            if h["k"] == _hRamp:
                _hlog.info(f"HOLD-L: L arm reached the low carry pose at step {self.step_count}")
        if h["pose"] is not None:
            out[7:14] = _th10.as_tensor(h["pose"], dtype=out.dtype)
            out[14] = -1.0
            if self.next_initial_actions is not None:
                self.next_initial_actions[:, 7:14] = h["pose"]; self.next_initial_actions[:, 14] = -1.0
        return out

    _hw.B1KPolicyWrapper.act, _hw.B1KPolicyWrapper.reset = _h_act, _h_reset
    print(f"serve_indep: hold-L-until-R rule on (task {_hTask}, after {_hN} holding steps, max {_hMax}, "
          f"{'pose ' + str(_hPose.round(3).tolist()) + f' ramp {_hRamp}' if _hPose is not None else 'pinned in place'})", flush=True)

if os.environ.get("B1K_CROUCH_TASK", "") != "":
    # Crouch-until-L-holds rule (added 2026-10-11, user, task 35): with joint_35_63_78 step 17500 the model stood up at
    # step 475, went for the tripod head first and did not bend to the floor camera until 7035 (7500 ckpt: first deep bend
    # 174-2013 in 16/16 runs). Proprio-only: while task == B1K_CROUCH_TASK, current_stage <= B1K_CROUCH_MAX_STAGE (1) and
    # the L hand has not held for B1K_CROUCH_STEPS (10) steps, the torso is not allowed to rise above the episode's start
    # crouch: torso joints 3:6 are replaced by the start pose whenever the model's torso0 is less bent (torso0 < start).
    # Deeper bends and the torso yaw (joint 6) keep the model's output. Ends for good once L holds, the stage passes
    # B1K_CROUCH_MAX_STAGE, or after B1K_CROUCH_MAX (5000) steps. Logs "CROUCH:".
    import logging as _lg15
    import numpy as _np15
    import torch as _th15
    from b1k.shared import eval_b1k_wrapper as _cw
    from b1k.policies.b1k_policy import extract_state_from_proprio as _esp15
    _crTask = int(os.environ["B1K_CROUCH_TASK"]); _crN = int(os.environ.get("B1K_CROUCH_STEPS", "10"))
    _crMaxStage = int(os.environ.get("B1K_CROUCH_MAX_STAGE", "1")); _crMax = int(os.environ.get("B1K_CROUCH_MAX", "5000"))
    _clog = _lg15.getLogger("b1k")
    _orig_act_c, _orig_reset_c = _cw.B1KPolicyWrapper.act, _cw.B1KPolicyWrapper.reset

    def _c_init(self):
        self._cr = {"pose": None, "l": 0, "n": 0, "hits": 0, "done": False}

    def _c_reset(self, *a, **kw):
        _c_init(self)
        return _orig_reset_c(self, *a, **kw)

    def _c_act(self, obs):
        if not hasattr(self, "_cr"):
            _c_init(self)
        c = self._cr
        st = _np15.asarray(_esp15(_np15.asarray(obs["robot_r1::proprio"]))).reshape(-1)
        out = _orig_act_c(self, obs)
        if self.task_id != _crTask or c["done"]:
            return out
        if c["pose"] is None:
            c["pose"] = st[3:6].astype(_np15.float32).copy()
            _clog.info(f"CROUCH: start torso {c['pose'].round(2).tolist()} at step {self.step_count}")
        c["n"] += 1
        c["l"] = c["l"] + 1 if (-0.9 < st[14] < 0.9 and float(out[14]) < 0) else 0
        why = ("L holding" if c["l"] >= _crN else f"stage {self.current_stage}" if int(self.current_stage) > _crMaxStage
               else "max duration" if c["n"] > _crMax else None)
        if why:
            c["done"] = True
            _clog.info(f"CROUCH: released at step {self.step_count} ({why}; {c['hits']} steps held down)")
            return out
        m0 = float(out[3])
        if m0 < float(c["pose"][0]):
            out[3:6] = _th15.as_tensor(c["pose"], dtype=out.dtype)
            if self.next_initial_actions is not None:
                self.next_initial_actions[:, 3:6] = c["pose"]
            c["hits"] += 1
            if c["hits"] in (1, 10, 100) or c["hits"] % 500 == 0:
                _clog.info(f"CROUCH: model torso0 {m0:.2f} -> kept at the start crouch at step {self.step_count} "
                           f"(#{c['hits']})")
        return out

    _cw.B1KPolicyWrapper.act, _cw.B1KPolicyWrapper.reset = _c_act, _c_reset
    print(f"serve_indep: crouch-until-L-holds rule on (task {_crTask}, stages <= {_crMaxStage}, L {_crN} steps, max {_crMax})",
          flush=True)

if os.environ.get("B1K_STAGE_FLOOR_TASK", "") != "":
    # Stage-floor rule (added 2026-10-11, task 35): the stage vote keeps flipping "pick up camera" (1) back to "move to
    # camera" (0), which restarts the approach; the pickup then stalls for thousands of steps. B1K_STAGE_FLOOR="a:b,..."
    # means: once stage >= a has been reached, current_stage never drops below b (e.g. "1:1" blocks 1 -> 0 only).
    import logging as _lg11
    from b1k.shared import eval_b1k_wrapper as _fw
    _fTask = int(os.environ["B1K_STAGE_FLOOR_TASK"])
    _fMap = [tuple(int(v) for v in kv.split(":")) for kv in os.environ.get("B1K_STAGE_FLOOR", "1:1").split(",") if kv]
    _flog = _lg11.getLogger("b1k")
    _orig_act_f, _orig_reset_f = _fw.B1KPolicyWrapper.act, _fw.B1KPolicyWrapper.reset

    def _f_reset(self, *a, **kw):
        self._fl = {"max": 0, "n": 0}
        return _orig_reset_f(self, *a, **kw)

    def _f_act(self, obs):
        if not hasattr(self, "_fl"):
            self._fl = {"max": 0, "n": 0}
        out = _orig_act_f(self, obs)
        if self.task_id != _fTask:
            return out
        self._fl["max"] = max(self._fl["max"], int(self.current_stage))
        floor = max([b for a, b in _fMap if self._fl["max"] >= a] or [0])
        if int(self.current_stage) < floor:
            self._fl["n"] += 1
            if self._fl["n"] <= 20 or self._fl["n"] % 100 == 0:
                _flog.info(f"STAGE-FLOOR: vote went {self.current_stage} at step {self.step_count} -> held at {floor} "
                           f"(#{self._fl['n']})")
            self.current_stage = floor
        return out

    _fw.B1KPolicyWrapper.act, _fw.B1KPolicyWrapper.reset = _f_act, _f_reset
    print(f"serve_indep: stage-floor rule on (task {_fTask}, {_fMap})", flush=True)

if os.environ.get("B1K_CLOSE_FREEZE_TASK", "") != "":
    # Close-freeze rule (added 2026-10-11, task 80): many task-80 closes end fully shut on nothing next to a thin glass
    # shard; the model starts closing while the hand is still moving. When a gripper command goes from open (>= 0) to
    # closing (< 0), that hand's arm joints (and, with B1K_CLOSE_FREEZE_BODY=1, torso joints + zero base velocity) are
    # pinned at their measured values for B1K_CLOSE_FREEZE_N steps while the fingers close. Logs "CLOSE-FREEZE".
    import logging as _lg12
    import numpy as _np12
    import torch as _th12
    from b1k.shared import eval_b1k_wrapper as _cw
    from b1k.policies.b1k_policy import extract_state_from_proprio as _esp12
    _cTask = int(os.environ["B1K_CLOSE_FREEZE_TASK"]); _cN = int(os.environ.get("B1K_CLOSE_FREEZE_N", "12"))
    _cBody = os.environ.get("B1K_CLOSE_FREEZE_BODY", "1") == "1"
    _clog = _lg12.getLogger("b1k")
    _orig_act_c, _orig_reset_c = _cw.B1KPolicyWrapper.act, _cw.B1KPolicyWrapper.reset
    _HANDS = {"L": (14, slice(7, 14)), "R": (22, slice(15, 22))}

    def _c_init(self):
        self._cf = {"prev": {"L": 1.0, "R": 1.0}, "left": {"L": 0, "R": 0}, "pose": {"L": None, "R": None},
                    "body": None, "n": 0}

    def _c_reset(self, *a, **kw):
        _c_init(self)
        return _orig_reset_c(self, *a, **kw)

    def _c_act(self, obs):
        if not hasattr(self, "_cf"):
            _c_init(self)
        cf = self._cf
        st = _np12.asarray(_esp12(_np12.asarray(obs["robot_r1::proprio"]))).reshape(-1)
        out = _orig_act_c(self, obs)
        if self.task_id != _cTask:
            return out
        for h, (g, arm) in _HANDS.items():
            cmd = float(out[g])
            if cf["prev"][h] >= 0 and cmd < 0 and cf["left"][h] == 0 and st[g] > 0.5:
                cf["left"][h] = _cN; cf["pose"][h] = st[arm].astype(_np12.float32).copy(); cf["n"] += 1
                if _cBody:
                    cf["body"] = st[3:7].astype(_np12.float32).copy()
                if cf["n"] <= 30 or cf["n"] % 20 == 0:
                    _clog.info(f"CLOSE-FREEZE: {h} close starts at step {self.step_count} (width {st[g]:+.2f}) -> holding "
                               f"{h} arm{' + body' if _cBody else ''} for {_cN} steps (#{cf['n']})")
            cf["prev"][h] = cmd
            if cf["left"][h] > 0:
                cf["left"][h] -= 1
                out[arm] = _th12.as_tensor(cf["pose"][h], dtype=out.dtype)
                if self.next_initial_actions is not None:
                    self.next_initial_actions[:, arm] = cf["pose"][h]
        if _cBody and (cf["left"]["L"] > 0 or cf["left"]["R"] > 0) and cf["body"] is not None:
            out[0:3] = 0.0
            out[3:7] = _th12.as_tensor(cf["body"], dtype=out.dtype)
            if self.next_initial_actions is not None:
                self.next_initial_actions[:, 0:3] = 0.0; self.next_initial_actions[:, 3:7] = cf["body"]
        return out

    _cw.B1KPolicyWrapper.act, _cw.B1KPolicyWrapper.reset = _c_act, _c_reset
    print(f"serve_indep: close-freeze rule on (task {_cTask}, {_cN} steps, body {_cBody})", flush=True)

if os.environ.get("B1K_SLOW_CLOSE_TASK", "") != "":
    # Slow-after-close rule (added 2026-10-14, user: "is there any way to prevent the tipping of the radio"). Task-0 public
    # passes: 14 radio tip-overs, ~11 right after a grasp. Signature: the fingers shut fully (-1.00, empty) within 2 steps,
    # then the model yanks the hand up ~16 cm in 10 steps; the closed fingers catch the radio's handle/edge and flip it,
    # and the model reopens 24-40 steps later. Good grasps rise only 1-3 cm per 10 steps. When a gripper command goes from
    # open to closing, for B1K_SLOW_CLOSE_N (60) steps that hand's arm joints and the torso may move at most
    # B1K_SLOW_CLOSE_RATE (0.008 rad) per step away from their measured values, and base speed is capped at 0.05.
    # Direction still comes from the model. Proprio only. Logs "SLOW-CLOSE".
    import logging as _lg21
    import numpy as _np21
    import torch as _th21
    from b1k.shared import eval_b1k_wrapper as _scw
    from b1k.policies.b1k_policy import extract_state_from_proprio as _esp21
    _scTask = int(os.environ["B1K_SLOW_CLOSE_TASK"]); _scN = int(os.environ.get("B1K_SLOW_CLOSE_N", "60"))
    _scRate = float(os.environ.get("B1K_SLOW_CLOSE_RATE", "0.008"))
    _sclog = _lg21.getLogger("b1k")
    _orig_act_sc, _orig_reset_sc = _scw.B1KPolicyWrapper.act, _scw.B1KPolicyWrapper.reset
    _SCH = {"L": (14, slice(7, 14)), "R": (22, slice(15, 22))}

    def _sc_init(self):
        self._sc = {"prev": {"L": 1.0, "R": 1.0}, "left": {"L": 0, "R": 0}, "n": 0, "clip": 0}

    def _sc_reset(self, *a, **kw):
        _sc_init(self)
        return _orig_reset_sc(self, *a, **kw)

    def _sc_act(self, obs):
        if not hasattr(self, "_sc"):
            _sc_init(self)
        c = self._sc
        st = _np21.asarray(_esp21(_np21.asarray(obs["robot_r1::proprio"]))).reshape(-1)
        out = _orig_act_sc(self, obs)
        if self.task_id != _scTask:
            return out
        for h, (g, arm) in _SCH.items():
            cmd = float(out[g])
            if c["prev"][h] >= 0 and cmd < 0 and st[g] > 0.5:
                c["left"][h] = _scN; c["n"] += 1
                if c["n"] <= 30 or c["n"] % 20 == 0:
                    _sclog.info(f"SLOW-CLOSE: {h} close starts at step {self.step_count} (width {st[g]:+.2f}) -> {h} arm + "
                                f"torso rate-limited to {_scRate} rad/step for {_scN} steps (#{c['n']})")
            c["prev"][h] = cmd
            if c["left"][h] > 0:
                c["left"][h] -= 1
                m = out[arm].detach().cpu().numpy() if hasattr(out[arm], "detach") else _np21.asarray(out[arm])
                lim = _np21.clip(m, st[arm] - _scRate, st[arm] + _scRate)
                if _np21.abs(lim - m).max() > 1e-4:
                    c["clip"] += 1
                out[arm] = _th21.as_tensor(lim.astype(_np21.float32), dtype=out.dtype)
        if c["left"]["L"] > 0 or c["left"]["R"] > 0:
            m = out[3:7].detach().cpu().numpy() if hasattr(out[3:7], "detach") else _np21.asarray(out[3:7])
            out[3:7] = _th21.as_tensor(_np21.clip(m, st[3:7] - _scRate, st[3:7] + _scRate).astype(_np21.float32),
                                       dtype=out.dtype)
            out[0:3] = _th21.clamp(out[0:3], -0.05, 0.05)
            if c["clip"] in (1, 100) or (c["clip"] > 0 and c["clip"] % 1000 == 0):
                _sclog.info(f"SLOW-CLOSE: model motion clipped at step {self.step_count} (#{c['clip']})")
        return out

    _scw.B1KPolicyWrapper.act, _scw.B1KPolicyWrapper.reset = _sc_act, _sc_reset
    print(f"serve_indep: slow-close rule on (task {_scTask}, {_scN} steps, {_scRate} rad/step)", flush=True)

if os.environ.get("B1K_STAGE_GAIN_TASK", "") != "":
    # Stage base-gain rule (added 2026-10-10, user request for task 35): in the given stage(s) the model shuffles the base
    # around the tripod for ~3000 steps before the R hand grips it (demos: ~470 steps from camera-in-hand to tripod hold).
    # While task == B1K_STAGE_GAIN_TASK, current_stage is in B1K_STAGE_GAIN_STAGES and hand B1K_STAGE_GAIN_HAND (default R)
    # is not yet holding, the model's own base-velocity commands (out[0:3]) are multiplied by B1K_STAGE_GAIN, so the base
    # goes where the model already wants, faster. Off as soon as that hand holds (width in (-0.9, 0.9)).
    import logging as _lg8
    import numpy as _np9
    from b1k.shared import eval_b1k_wrapper as _gw
    from b1k.policies.b1k_policy import extract_state_from_proprio as _esp8
    _gTask = int(os.environ["B1K_STAGE_GAIN_TASK"])
    _gStages = {int(x) for x in os.environ.get("B1K_STAGE_GAIN_STAGES", "3").split(",")}
    _gGain = float(os.environ.get("B1K_STAGE_GAIN", "2.0"))
    # 2026-10-10: optional per-stage map "stage:gain,..." (e.g. "0:0.5,1:0.5,2:2.0": slow the base while going for the
    # camera so it stops shoving it along the floor, fast on the walk to the tripod). With a map there is no hand check:
    # the stage alone decides, and stages not in the map are untouched.
    _gMap = {int(k): float(v) for k, v in (kv.split(":") for kv in os.environ.get("B1K_STAGE_GAIN_MAP", "").split(",") if kv)}
    _gg = 22 if os.environ.get("B1K_STAGE_GAIN_HAND", "R").upper() == "R" else 14
    _glog = _lg8.getLogger("b1k")
    _orig_act_g, _orig_reset_g = _gw.B1KPolicyWrapper.act, _gw.B1KPolicyWrapper.reset

    def _g_reset(self, *a, **kw):
        self._gs = {"on": False, "n": 0}
        return _orig_reset_g(self, *a, **kw)

    def _g_act(self, obs):
        if not hasattr(self, "_gs"):
            self._gs = {"on": False, "n": 0}
        st = _np9.asarray(_esp8(_np9.asarray(obs["robot_r1::proprio"]))).reshape(-1)
        out = _orig_act_g(self, obs)
        if self.task_id != _gTask:
            return out
        if _gMap:
            g = _gMap.get(int(self.current_stage), 1.0)
            if g != self._gs.get("g", 1.0):
                _glog.info(f"STAGE-GAIN: gain {g} at step {self.step_count} (stage {self.current_stage})")
                self._gs["g"] = g
            if g != 1.0:
                out[0:3] = out[0:3] * g
            return out
        on = self.current_stage in _gStages and not (-0.9 < st[_gg] < 0.9)
        if on != self._gs["on"]:
            _glog.info(f"STAGE-GAIN: {'ON' if on else 'off'} at step {self.step_count} (stage {self.current_stage}, "
                       f"hand width {st[_gg]:+.2f}, gain {_gGain})")
            self._gs["on"] = on
        if on:
            out[0:3] = out[0:3] * _gGain
            self._gs["n"] += 1
        return out

    _gw.B1KPolicyWrapper.act, _gw.B1KPolicyWrapper.reset = _g_act, _g_reset
    print(f"serve_indep: stage base-gain rule on (task {_gTask}, "
          f"{'map ' + str(_gMap) if _gMap else f'stages {sorted(_gStages)}, gain {_gGain}'})", flush=True)

if os.environ.get("B1K_GRIP_DEBOUNCE", "") != "":
    # Gripper debounce (added 2026-10-11, task 80): at chunk starts the model often emits 1-3-step half-closes
    # (cmd -0.03..-0.5) that reopen right away, i.e. the visible finger twitching (inst 1 close-freeze run: 77 R close
    # starts, 53 of them <= 5 steps). A close command only passes once the model has asked for it (cmd < 0) for
    # B1K_GRIP_DEBOUNCE consecutive steps; until then that hand gets +1 (open). Per hand, every task. Logs "DEBOUNCE".
    import logging as _lg13
    from b1k.shared import eval_b1k_wrapper as _dw
    _dK = int(os.environ["B1K_GRIP_DEBOUNCE"])
    _dlog = _lg13.getLogger("b1k")
    _orig_act_d, _orig_reset_d = _dw.B1KPolicyWrapper.act, _dw.B1KPolicyWrapper.reset

    def _d_reset(self, *a, **kw):
        self._db = {14: 0, 22: 0, "n": 0}
        return _orig_reset_d(self, *a, **kw)

    def _d_act(self, obs):
        if not hasattr(self, "_db"):
            self._db = {14: 0, 22: 0, "n": 0}
        out = _orig_act_d(self, obs)
        for g in (14, 22):
            self._db[g] = self._db[g] + 1 if out[g] < 0 else 0
            if 0 < self._db[g] < _dK:
                out[g] = 1.0
            elif self._db[g] == 0 and 0 < self._db.get(("run", g), 0) < _dK:
                self._db["n"] += 1
                if self._db["n"] <= 20 or self._db["n"] % 50 == 0:
                    _dlog.info(f"DEBOUNCE: {'L' if g == 14 else 'R'} {self._db[('run', g)]}-step close blip dropped "
                               f"at step {self.step_count} (#{self._db['n']})")
            self._db[("run", g)] = self._db[g]
        return out

    _dw.B1KPolicyWrapper.act, _dw.B1KPolicyWrapper.reset = _d_act, _d_reset
    print(f"serve_indep: gripper debounce on (close passes after {_dK} consecutive close steps)", flush=True)

if os.environ.get("B1K_GLASS_GATE_TASK", "") != "":
    # Task-80 stage gate (added 2026-10-11, user request). Demo skill order (annotations task-0080): 0 move to glass,
    # 1 pick up, 2 pick up, 3 move to trash can, 4 place in, 5 place in, 6 move to glass, 7 pick up, 8 move to trash can,
    # 9 place in. Gate A: before the first delivery (stage 4 never reached), stage 3 ("move to trash can") is only
    # allowed while BOTH hands hold something, and a jump to 4-7 without having passed gate A is held at 2.
    # Gate B: stage 8 ("move to trash can" with the last glass) is only allowed while at least one hand holds
    # something, and a jump to 9 without having passed gate B is held at 7. "Holding" = width in (-0.9, 0.9) while
    # commanded closed, for >= B1K_GLASS_GATE_STEPS (default 10) consecutive steps. Logs "GLASS-GATE".
    import logging as _lg14
    import numpy as _np14
    from b1k.shared import eval_b1k_wrapper as _gw
    from b1k.policies.b1k_policy import extract_state_from_proprio as _esp14
    _ggTask = int(os.environ["B1K_GLASS_GATE_TASK"]); _ggN = int(os.environ.get("B1K_GLASS_GATE_STEPS", "10"))
    _gglog = _lg14.getLogger("b1k")
    _orig_act_gg, _orig_reset_gg = _gw.B1KPolicyWrapper.act, _gw.B1KPolicyWrapper.reset

    def _gg_init(self):
        self._gg = {"h": {14: 0, 22: 0}, "okA": False, "okB": False, "deliv": False, "n": 0}

    def _gg_reset(self, *a, **kw):
        _gg_init(self)
        return _orig_reset_gg(self, *a, **kw)

    def _gg_act(self, obs):
        if not hasattr(self, "_gg"):
            _gg_init(self)
        st = _np14.asarray(_esp14(_np14.asarray(obs["robot_r1::proprio"]))).reshape(-1)
        out = _orig_act_gg(self, obs)
        if self.task_id != _ggTask:
            return out
        g = self._gg
        for k in (14, 22):
            g["h"][k] = g["h"][k] + 1 if (-0.9 < st[k] < 0.9 and out[k] < 0) else 0
        n_hold = sum(g["h"][k] >= _ggN for k in (14, 22))
        s = int(self.current_stage); cap = None
        if s >= 4 and g["okA"]:
            g["deliv"] = True
        if not g["deliv"]:
            if s == 3:
                if n_hold == 2:
                    g["okA"] = True
                else:
                    cap = 2
            elif 4 <= s <= 7 and not g["okA"]:
                cap = 2
            elif s >= 8 and not g["okA"]:
                cap = 2
        else:
            if s == 8:
                if n_hold >= 1:
                    g["okB"] = True
                else:
                    cap = 7
            elif s == 9 and not g["okB"]:
                cap = 7
        if cap is not None:
            g["n"] += 1
            if g["n"] <= 20 or g["n"] % 100 == 0:
                _gglog.info(f"GLASS-GATE: stage {s} at step {self.step_count} with {n_hold} hand(s) holding -> held at "
                            f"{cap} (#{g['n']})")
            self.current_stage = cap
        return out

    _gw.B1KPolicyWrapper.act, _gw.B1KPolicyWrapper.reset = _gg_act, _gg_reset
    print(f"serve_indep: glass stage gate on (task {_ggTask}, holding >= {_ggN} steps)", flush=True)

if os.environ.get("B1K_TRACE_DIR", "") != "":
    # Per-step gripper trace (added 2026-10-04): commanded vs measured gripper, both in the normalized [-1 closed, 1 open]
    # space (action/state idx 14 = left, 22 = right). One CSV per episode: <dir>/trace_epN.csv.
    import numpy as _np
    from b1k.shared import eval_b1k_wrapper as _tw
    from b1k.policies.b1k_policy import extract_state_from_proprio as _esp
    _tdir = os.environ["B1K_TRACE_DIR"]; os.makedirs(_tdir, exist_ok=True)
    _tr = {"ep": 0, "f": None, "n": 0}
    _orig_reset, _orig_act = _tw.B1KPolicyWrapper.reset, _tw.B1KPolicyWrapper.act
    def _t_reset(self, *a, **kw):
        if _tr["f"] is not None and _tr["n"] > 0:
            _tr["f"].close(); _tr["f"] = None
        return _orig_reset(self, *a, **kw)
    def _t_act(self, obs):
        if _tr["f"] is None:
            _tr["ep"] += 1; _tr["n"] = 0
            _tr["f"] = open(os.path.join(_tdir, f"trace_ep{_tr['ep']}.csv"), "w")
            _tr["f"].write("step,stage,new_chunk,cmd_L,cmd_R,meas_L,meas_R,chunk_min_L,chunk_max_L,chunk_min_R,chunk_max_R,cmd_bx,cmd_by,cmd_bw,torso0,torso1,torso2,torso3\n")
        new = self.last_actions is None or self.action_index >= self.config.execute_in_n_steps
        st = _esp(_np.asarray(obs["robot_r1::proprio"]))
        st = _np.asarray(st).reshape(-1)
        out = _orig_act(self, obs)
        a = out.numpy() if hasattr(out, "numpy") else _np.asarray(out)
        ch = self.last_actions
        _tr["f"].write(f"{self.step_count},{self.current_stage},{int(new)},{a[14]:.4f},{a[22]:.4f},{st[14]:.4f},{st[22]:.4f},"
                       f"{ch[:,14].min():.4f},{ch[:,14].max():.4f},{ch[:,22].min():.4f},{ch[:,22].max():.4f},"
                       f"{a[0]:.4f},{a[1]:.4f},{a[2]:.4f},{st[3]:.4f},{st[4]:.4f},{st[5]:.4f},{st[6]:.4f}\n")
        _tr["n"] += 1
        if _tr["n"] % 100 == 0: _tr["f"].flush()
        return out
    _tw.B1KPolicyWrapper.reset, _tw.B1KPolicyWrapper.act = _t_reset, _t_act
    print(f"serve_indep: gripper trace -> {_tdir}", flush=True)

if os.environ.get("B1K_STAGE_FILE", "") != "":
    # Stage publisher (added 2026-10-14, task-0 PRESS-ASSIST): after every act() the wrapper's current stage is written to
    # B1K_STAGE_FILE as "<step> <stage>" so the eval client's press rule knows when the model is in its press stage.
    # Only the policy's own internal stage is shared (no sim information).
    from b1k.shared import eval_b1k_wrapper as _sw
    _sfile = os.environ["B1K_STAGE_FILE"]
    _orig_act_s = _sw.B1KPolicyWrapper.act
    def _s_act(self, obs):
        out = _orig_act_s(self, obs)
        try:
            with open(_sfile + ".tmp", "w") as _fh:
                _fh.write(f"{self.step_count} {int(self.current_stage)}")
            os.replace(_sfile + ".tmp", _sfile)
        except Exception:
            pass
        return out
    _sw.B1KPolicyWrapper.act = _s_act
    print(f"serve_indep: stage publisher -> {_sfile}", flush=True)

sys.argv[0] = os.path.join(REPO, "scripts", "serve_b1k.py")
runpy.run_path(sys.argv[0], run_name="__main__")
