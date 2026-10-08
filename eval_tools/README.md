# Eval / diagnostic tools (2026-10, eval pod)

Paths inside assume the eval pod layout (`/workspace/persistent/...`, checkpoints under `~/ckpt`, demos under `~/demos2026`).

| file | what |
|---|---|
| `serve_indep.py` | wraps `scripts/serve_b1k.py`; env switches: `B1K_INDEP_NOISE`, `B1K_CHAIN_76_80_87` (649-row stage table), `B1K_PIN_STAGE`, `B1K_NO_INPAINT`, `B1K_SEED`, `B1K_RELEASE_STAGE` (release rule), **`B1K_EMPTY_OPEN_N` / `B1K_EMPTY_THR`** (empty-gripper re-open rule), **`B1K_TRACE_DIR`** (per-step gripper cmd vs measured CSV) |
| `start_server_chain82.sh` | launches `serve_indep.py`; `SERVE_EXTRA` passes extra serve args (e.g. `--execute-in-n-steps 26` = no action compression) |
| `eval_hf_ckpt.sh`, `eval_hf_ckpt_tag.sh` | download an HF checkpoint folder, serve it, run `scripts/run_custom_eval.py` episodes (`TAG_SUFFIX` = separate output dir) |
| `run_chain82_subset.sh` | runs the sim episodes for a task/instance list |
| `gripper_trace_summary.py` | summarises a `B1K_TRACE_DIR` CSV (per-bin cmd/meas, open/close events and response) |
| `offline_probe_task76.py`, `offline_probe_generic.py` | feed recorded demo frames to a checkpoint; gripper open/close, base velocity and arm-motion agreement with the demo; `--prompt-task` swaps the task id |
| `probe_taskid.py` | task-id sensitivity: action change from swapping the task id vs sampling noise |
| `path_overlay.py` | eval base path (harness pose log) over the scene floor map + demo path (integrated `base_qvel`, start fitted to annotated grasp events) |
| `watch_hf_tasks.py`, `watch_hf_any_ready.py`, `watch_task0_commits.py`, `hf_changed_ckpts.py` | HF checkpoint watchers (READY-marker contract) |

`scripts/run_custom_eval.py` changes in this branch: tasks 80/87 registered; joint-count drift fix (`_dof_tolerant_load_state`, restaurant_diner cabinets); BDDL override for `clean_up_broken_glass` matching the 2026 template; **pose log** (`<log_path>/poses/<task>_<inst>.jsonl`: base x, y, yaw every 10 steps, task-object positions every 500).
