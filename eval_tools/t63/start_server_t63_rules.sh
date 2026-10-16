#!/bin/bash
# Task-63 policy server with eval-diagnostics-2026-10 server rules (RULES.md best public setup), CORRELATED noise.
source /workspace/persistent/env.sh
export B1K_INDEP_NOISE=0 B1K_CHAIN_35_63_78=1
export B1K_CLOSE_FREEZE_TASK=63 B1K_CLOSE_FREEZE_N=20 B1K_CLOSE_FREEZE_BODY=1
export B1K_GRIP_DEBOUNCE=4 B1K_EMPTY_OPEN_N=0 B1K_EMPTY_THR=-0.95
export B1K_TRACE_DIR=${B1K_TRACE_DIR:?}
CKPT=${CKPT:-$OV/ckpt/joint_35_63_78/joint_35_63_78_pickup_step27500}
cd $REPO
exec $OV/venvs/p4t/bin/python -u /workspace/persistent/diag/serve_indep_launch.py --port ${PORT:-8000} policy:checkpoint \
  --policy.config pi_behavior_b1k_fast_100task --policy.dir $CKPT
