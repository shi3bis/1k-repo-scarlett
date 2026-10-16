"""Per-instance task-63 summary from run_custom_eval logs + pose logs: grasp, wall arrival, sweep, closest attach approach."""
import json, re, sys, glob, os
O = sys.argv[1]
for f in sorted(glob.glob(f"{O}/poses/installing_smoke_detectors_*.jsonl")):
    inst = re.search(r"_(\d+)\.jsonl", f).group(1)
    att = [(r["t"],) + tuple(v[:2]) for l in open(f) for r in [json.loads(l)] if "att" in r for v in r["att"].values()]
    best = min(att, key=lambda a: a[1] + a[2] / 300) if att else None
    log = open(f"{O}/logs/inst{inst}.log", errors="ignore").read() if os.path.exists(f"{O}/logs/inst{inst}.log") else ""
    g = re.search(r"GRASP-DESCEND: (\w+) closed at step (\d+)", log)
    w = re.search(r"WALL-HOLD: (\w+) open at the wall at step (\d+)", log)
    s = re.search(r"WALL-TWIST: all (\d+) tries done at step (\d+)", log)
    m = f"{O}/metrics/installing_smoke_detectors_{inst}_0.json"
    q = json.load(open(m))["q_score"]["final"] if os.path.exists(m) else None
    print(f"{inst}: q={q} grasp={g.groups() if g else None} wall={w.groups() if w else None} sweep_done={s.groups() if s else None} "
          f"closest(t,dist,deg)={tuple(round(x,3) for x in best) if best else None} final={tuple(round(x,3) for x in att[-1]) if att else None}")
