#!/bin/bash
# Policy server for the chain82 ckpt (independent noise) on :8000. Log: logs/policy_server_chain82.log
# 2026-10-06: optional SERVE_EXTRA (e.g. "--execute-in-n-steps 26" = no action compression / speed-up off).
source /workspace/persistent/env.sh
export PATH=$HOME/.local/bin:$PATH
export UV_PROJECT_ENVIRONMENT=/home/kasm-user/venvs/p4t VIRTUAL_ENV=/home/kasm-user/venvs/p4t
export TORCHDYNAMO_DISABLE=1 B1K_INDEP_NOISE=${B1K_INDEP_NOISE:-1}  # caller may set 0 (corr noise ckpts)
CKPT=${CKPT:-/home/kasm-user/ckpt/hf/iliastats_chain/task82_store_batteries}
cd /workspace/persistent/repos/place_inside_4_tasks
exec uv run --no-sync python /workspace/persistent/diag/serve_indep.py --port 8000 ${SERVE_EXTRA:-} policy:checkpoint \
  --policy.config ${CONFIG:-pi_behavior_b1k_fast_100task} --policy.dir "$CKPT"
