#!/bin/bash
# [copy of eval_hf_ckpt.sh with optional env TAG_SUFFIX appended to the output/log tag, 2026-10-04]
# Evaluate one checkpoint revision from an HF model repo in the sim.
#   eval_hf_ckpt.sh REPO REVISION TASK_NAME "INSTANCES" MAX_STEPS [CONFIG] [SUBDIR]
# SUBDIR: checkpoint folder inside the repo when it is not at the root (e.g. lowlr_decay -> lowlr_decay/params,
# lowlr_decay/assets). NOTE 2026-10-03: without it, commit ecfa453 evaluated the ROOT params = the old 3569c80 model.
# e.g. eval_hf_ckpt.sh shi2bis/task35_attach_a_camera_to_a_tripod 3569c8022f attach_a_camera_to_a_tripod "1 2" 7823
# Downloads REPO@REVISION (params/ + assets/ at the repo root) to ~/ckpt/<repo>/<rev7> (overlay), serves it with
# the default settings (independent noise, eval tricks on, normal stage tracking) on :8000, runs the episodes,
# stops the server. Waits first if another sim eval / policy server is already running (one GPU, one port).
# Output: /workspace/persistent/eval_outputs_<repo name>/<rev7>/{metrics,videos,logs,progress.txt}
set -u
REPO=$1; REV=$2; TASK=$3; INST=$4; MAXS=$5; CONFIG=${6:-pi_behavior_b1k_fast_100task}; SUB=${7:-}
R=/workspace/persistent
NAME=${REPO#*/}; REV7=${REV:0:7}; TAG=$REV7${SUB:+_$SUB}${TAG_SUFFIX:+_$TAG_SUFFIX}
DL=/home/kasm-user/ckpt/$NAME/$REV7          # repo snapshot root
CK=$DL${SUB:+/$SUB}                           # folder holding params/ + assets/
OUT=$R/eval_outputs_$NAME/$TAG
LOG=$R/logs/eval_hf_${NAME}_$TAG.log
mkdir -p $OUT
exec >> $LOG 2>&1
echo "$(date -u) eval_hf_ckpt $REPO@$REV subdir=${SUB:-<root>} task=$TASK instances=$INST max_steps=$MAXS config=$CONFIG"

# one sim eval at a time. Patterns are anchored to the real eval/server command lines (2026-10-11): a plain
# "[r]un_custom_eval.py" match also hit idle launcher shells whose command text merely mentioned the file name.
while pgrep -f "^python -u scripts/[r]un_custom_eval[.]py" >/dev/null || pgrep -f "^(uv run --no-sync python|/home/kasm-user/venvs/p4t/bin/python) [^ ]*[s]erve_indep[.]py --port" >/dev/null; do
  echo "$(date -u) another eval/server is running -- waiting"; sleep 120
done

source $R/env.sh
if [ ! -f $CK/params/manifest.ocdbt ]; then
  hf download $REPO --revision $REV --include "${SUB:+$SUB/}params/*" --include "${SUB:+$SUB/}assets/*" --local-dir $DL || { echo "DOWNLOAD FAILED"; exit 1; }
fi
echo "norm_stats md5: $(md5sum $CK/assets/*/*/norm_stats.json | cut -c1-32)  params manifest md5: $(md5sum $CK/params/manifest.ocdbt | cut -c1-32)"

CONFIG=$CONFIG CKPT=$CK setsid $R/diag/start_server_chain82.sh > $R/logs/policy_server_${NAME}_$TAG.log 2>&1 &
SP=$!
until grep -qE "server listening|Traceback" $R/logs/policy_server_${NAME}_$TAG.log || ! kill -0 $SP 2>/dev/null; do sleep 5; done
if ! grep -q "server listening" $R/logs/policy_server_${NAME}_$TAG.log; then
  echo "$(date -u) SERVER FAILED (see logs/policy_server_${NAME}_$TAG.log)"; kill -- -$SP 2>/dev/null; exit 1
fi
OUT=$OUT TASKS=$TASK INSTANCES="$INST" MAX_STEPS=$MAXS $R/diag/run_chain82_subset.sh
kill -- -$SP 2>/dev/null; sleep 10
echo "$(date -u) DONE $REPO@$TAG"
python3 - "$OUT" <<'EOF'
import json, glob, sys
for f in sorted(glob.glob(sys.argv[1] + "/metrics/*.json")):
    d = json.load(open(f))
    print(f"RESULT {d['task']} inst {d['instance_id']}: q={d['q_score']['final']:.3f} success={d['success']} steps={d['steps']}")
EOF
