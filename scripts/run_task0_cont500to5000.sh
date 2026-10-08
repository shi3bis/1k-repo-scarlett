#!/bin/bash
# Task 0 continuation, restart (2026-10-07 evening). The 06:10 run crashed at its step-1000
# save (/workspace full: overlay redirect symlink silently landed INSIDE an existing dir).
# Resume from that run's step-500 save: 4500 more steps (~9 h) -> 5000 continuation total
# (7000 from ext100). All saves on /workspace (persistent), no overlay redirect.
# Then eval + push to public HF repo shi2bis/task0_turning_on_radio_cont5000 with READY.
set -x
source /workspace/persistent/behavior/env.sh >/dev/null 2>&1
conda activate behavior
B=/workspace/persistent/behavior; WT0=$B/wt-task0
P=$WT0/outputs/checkpoints/pi_behavior_b1k_fast_100task
INIT=$P/task0_radio_cont5000/500/params
EXP=task0_radio_cont500to5000
[ -d $INIT ] || { echo "FATAL: init $INIT missing"; exit 1; }
FREE=$(df --output=avail -BG /workspace | tail -1 | tr -dc 0-9)
[ "$FREE" -ge 26 ] || { echo "FATAL: only ${FREE}G free on /workspace, need >=26G (2 saves x 12G)"; exit 1; }
cd $WT0
PYTHONPATH=$WT0/src python -u scripts/train_task0.py --init-ckpt $INIT \
  --init-label "task0 cont step 500 (2500 total)" --exp-name $EXP --num-steps 4500 --no-push
RC=$?
echo "$(date -Is) train exited rc=$RC"
[ $RC -eq 0 ] || { echo "FATAL: training failed"; exit 1; }
D=$(ls -d $P/$EXP/*/ | grep -v tmp | sort -V | tail -1); D=${D%/}
[ -d "$D/params" ] || { echo "FATAL: no final checkpoint"; exit 1; }
echo "FINAL: $D"
PYTHONPATH=$WT0/src python -u scripts/eval_push.py --checkpoint-dir $D --label task0_cont5000 --own-task 0 || echo "$(date -Is) eval FAILED"
PYTHONPATH=$WT0/src:$WT0/scripts python -u - "$D" <<'PY'
import json, sys
from pathlib import Path
from huggingface_hub import HfApi
import train_task0 as t
D = Path(sys.argv[1])
t.HF_REPO = "shi2bis/task0_turning_on_radio_cont5000"
t.HF_FOLDER = "task0_turning_on_radio_cont5000"
t.NUM_STEPS = 5000
t.RECIPE = (f"5000 steps from the task0 2000-step final (7000 total from ext100; run as 500 + 4500 after a crash), "
            f"bs {t.BATCH_SIZE}, corr noise, Ilia stats, LR warmup {t.WARMUP} -> {t.PEAK_LR:g} cosine -> {t.END_LR:g} (restarted at 500)")
api = HfApi()
api.create_repo(t.HF_REPO, private=False, exist_ok=True)
t.push(str(D / "params"))  # params/ + assets/, verify, then READY
recs = [json.loads(l) for l in open("/workspace/persistent/behavior/logs/push_evals_task0.jsonl")]
recs = [r for r in recs if r["label"] == "task0_cont5000" and r["checkpoint_dir"] == str(D)]
if recs:
    api.upload_file(path_or_fileobj=json.dumps(recs[-1], indent=1).encode(), repo_id=t.HF_REPO,
                    path_in_repo=f"{t.HF_FOLDER}/eval_metrics.json", commit_message="offline eval metrics")
    print("uploaded eval_metrics.json", flush=True)
PY
echo "$(date -Is) push rc=$? ; done"
