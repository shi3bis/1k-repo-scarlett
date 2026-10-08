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
    _elog = _lg2.getLogger("b1k")
    _orig_act_e, _orig_reset_e = _ew.B1KPolicyWrapper.act, _ew.B1KPolicyWrapper.reset
    def _e_reset(self, *a, **kw):
        self._eo = {14: [0, False], 22: [0, False]}
        return _orig_reset_e(self, *a, **kw)
    def _e_act(self, obs):
        if not hasattr(self, "_eo"):
            self._eo = {14: [0, False], 22: [0, False]}
        st = _np3.asarray(_esp2(_np3.asarray(obs["robot_r1::proprio"]))).reshape(-1)
        out = _orig_act_e(self, obs)
        for g, name in ((14, "L"), (22, "R")):
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
            if c[1]:
                out[g] = 1.0
                if self.next_initial_actions is not None:
                    self.next_initial_actions[:, g] = 1.0
        return out
    _ew.B1KPolicyWrapper.act, _ew.B1KPolicyWrapper.reset = _e_act, _e_reset
    print(f"serve_indep: empty-gripper re-open rule on (N={_eN} steps, empty if width < {_eT})", flush=True)

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
            _tr["f"].write("step,stage,new_chunk,cmd_L,cmd_R,meas_L,meas_R,chunk_min_L,chunk_max_L,chunk_min_R,chunk_max_R\n")
        new = self.last_actions is None or self.action_index >= self.config.execute_in_n_steps
        st = _esp(_np.asarray(obs["robot_r1::proprio"]))
        st = _np.asarray(st).reshape(-1)
        out = _orig_act(self, obs)
        a = out.numpy() if hasattr(out, "numpy") else _np.asarray(out)
        ch = self.last_actions
        _tr["f"].write(f"{self.step_count},{self.current_stage},{int(new)},{a[14]:.4f},{a[22]:.4f},{st[14]:.4f},{st[22]:.4f},"
                       f"{ch[:,14].min():.4f},{ch[:,14].max():.4f},{ch[:,22].min():.4f},{ch[:,22].max():.4f}\n")
        _tr["n"] += 1
        if _tr["n"] % 100 == 0: _tr["f"].flush()
        return out
    _tw.B1KPolicyWrapper.reset, _tw.B1KPolicyWrapper.act = _t_reset, _t_act
    print(f"serve_indep: gripper trace -> {_tdir}", flush=True)

sys.argv[0] = os.path.join(REPO, "scripts", "serve_b1k.py")
runpy.run_path(sys.argv[0], run_name="__main__")
