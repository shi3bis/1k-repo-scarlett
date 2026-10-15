# Eval-time rules (BEHAVIOR 2026)

Every rule is off by default and switched on with environment variables. Client rules live in
`scripts/run_custom_eval.py` and wrap `Evaluator.step`. Server rules live in `eval_tools/serve_indep.py` and wrap
`B1KPolicyWrapper.act`. Each rule logs a tagged line (the tag in caps below).

All rules use only the robot's own state, its cameras and the policy's internal stage. The exceptions are the
pick-dot hold and the door rules, which read sim poses and are diagnostic only.

## Task 63 (installing_smoke_detectors). Best public setup: 0.30 on 301-310
| Env | Rule |
|---|---|
| `B1K_GRASP_DESCEND=1`, `B1K_GD_MAXZ`, `B1K_GD_DEPTH` | GRASP-DESCEND: on a close near the table, lower the open hand up to 3 cm (or to contact), then close with the arm held |
| `B1K_CLOSE_FREEZE_TASK=63`, `_N=20`, `_BODY=1` | CLOSE-FREEZE (server): freeze arm/torso/base for N steps while the fingers close |
| `B1K_GRIP_DEBOUNCE=4` | Debounce (server): a close passes only after 4 consecutive close commands |
| `B1K_EMPTY_OPEN_N=0`, `B1K_EMPTY_THR=-0.95`, `B1K_EMPTY_HANDS` | EMPTY-OPEN (server): reopen a hand whose fingers shut on nothing (`HANDS=R` limits it to the right hand) |
| `B1K_WALL_HOLD=1.70`, `B1K_WALL_EXACT=1` | WALL-HOLD: no release below 1.45 m; lift the hand to exactly 1.70 m before releasing |
| `B1K_WALL_YAWLOCK=1`, `_V`, `_VX` | YAW-LOCK: no base turning while the hand is high; sideways/forward speed caps |
| `B1K_WALL_DOT=1`, `_DOT_OFF=177`, `_DOT_OFFS`, `_DOT_LAT`, `B1K_WALL_DOT_RIGHT=1` | DOT-ALIGN: red-dot angle in the carrying wrist camera -> needed roll (fit `need = dot - 177`; see `dotcal.py`) |
| `B1K_WALL_TWIST=1`, `_FAST`, `_PRESS`, `_SEAT`, `_NOLOCK`, `_ZERO_FIRST`, `_ANGLES` | WALL-TWIST: back off / roll / press / seat / pull sweep over angles; snags are not counted as attached |
| `B1K_WALL_REACH=0.78`, `B1K_WALL_CREEP_PRESS=1` | Base creeps forward while pressing if the hand is > 0.78 m from the base |
| `B1K_WALL_PULL_BASE=1` | The base reverses during the pull-back (a stretched arm cannot retract on its own) |
| `B1K_SPLIT=public_test` | Run the official public test instances (301-320; 301-310 are reported) |

Tried and dropped: PICK-DOT / `B1K_GD_HOLDXY` (hurt grasps), `B1K_WALL_EARLY` (ALIGN-EARLY, 0/6),
`B1K_WALL_PRESS_CREEP` (no clear effect).

## Task 0 (turning_on_radio), ckpt task0_turning_on_radio_cont5000. Plain public: 0.50 / 0.20
| Env | Rule |
|---|---|
| `B1K_SLOW_CLOSE_TASK=0`, `_N=60`, `_RATE=0.008` | SLOW-CLOSE (server): for 60 steps after a close starts, arm and torso move at most 0.008 rad/step; base capped at 0.05. Stops the post-grasp yank that slid the handle out or flipped the radio. Partial pass: 4/8 |
| `B1K_STAGE_FILE=/dev/shm/...` | Stage publisher (server): writes the policy's current stage for the client rules |
| `B1K_T0_BTN=1`, `_WAIT=150`, `_TX/_TY=155/135`, `_AREA=450` | BUTTON-SERVO: visual servo of the L hand onto the red button (left wrist camera), then poke along the camera axis |
| `B1K_T0_PRESS=1` | PRESS-ASSIST v1: blind L-toward-R pokes. Harmful (0.10), dropped |
