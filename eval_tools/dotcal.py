"""Right-hand DOT-ALIGN calibration (2026-10-14). For every t63 pose-logged episode: carrying hand + carry step from the
log, arrival = first pose row after the carry with the detector < 12 cm short of the wall plane and above 1.2 m;
need = twist about world +x (the wall normal) taking the detector to the attach orientation (mean of the PULLBASE
successes, detector-in-nail frame); dot = red-pixel centroid in the carrying wrist's video panel (left: top-left,
right: bottom-left) on frames between carry+60 and arrival."""
import json, glob, os, re, sys
import numpy as np, cv2
from scipy.spatial.transform import Rotation as R
ROOT = "/workspace/persistent/eval_outputs_joint_35_63_78"
E = glob.glob(f"{ROOT}/0abd041*PULLBASE_RULESPASS*carriers")[0]
rels = []
for i in (303, 304, 307):
    d = [json.loads(l) for l in open(f"{E}/poses/installing_smoke_detectors_{i}.jsonl")][-1]
    rels.append(R.from_quat(d["objq"]["wall_nail.n.01_1"]).inv() * R.from_quat(d["objq"]["fire_alarm.n.02_1"]))
REL = R.concatenate(rels).mean()

def twist_x(Rw):
    rv = Rw.as_quat()  # x y z w
    t = 2 * np.arctan2(rv[0], rv[3])
    return (np.degrees(t) + 180) % 360 - 180

out = []
for p in sorted(glob.glob(f"{ROOT}/*/poses/installing_smoke_detectors_*.jsonl")):
    d = os.path.dirname(os.path.dirname(p)); i = re.findall(r"_(\d+)\.jsonl", p)[0]
    lg = f"{d}/logs/installing_smoke_detectors_{i}.log"; v = f"{d}/videos/installing_smoke_detectors_{i}_0.mp4"
    if not (os.path.exists(lg) and os.path.exists(v)):
        continue
    m = re.search(r"WALL-HOLD: (left|right) carrying .* at step (\d+)", open(lg, errors="ignore").read())
    if not m:
        continue
    hand, cs = m.group(1), int(m.group(2))
    rows = [json.loads(l) for l in open(p)]
    arr = None
    for r in rows:
        if r["t"] < cs or not r.get("att"): continue
        a = list(r["att"].values())[0]
        if len(a) > 2 and a[2][0] < 0.12 and r["objs"]["fire_alarm.n.02_1"][2] > 1.2:
            arr = r; break
    if arr is None:
        continue
    Rn = R.from_quat(arr["objq"]["wall_nail.n.01_1"]); Rd = R.from_quat(arr["objq"]["fire_alarm.n.02_1"])
    need = twist_x((Rn * REL) * Rd.inv())
    full = ((Rn * REL) * Rd.inv()).magnitude() * 180 / np.pi
    c = cv2.VideoCapture(v); pts = []
    for fr in range(cs + 60, arr["t"], 20):
        c.set(1, fr); ok, im = c.read()
        if not ok: break
        pan = im[0:224, 0:224] if hand == "left" else im[224:448, 0:224]
        b, g, rr = [pan[..., k].astype(int) for k in range(3)]
        mk = (rr > 140) & (g < 90) & (b < 90)
        ys, xs = np.nonzero(mk)
        if len(xs) >= 4: pts.append((xs.mean(), ys.mean()))
    if len(pts) < 3:
        continue
    pts = np.array(pts)
    out.append(dict(run=os.path.basename(d)[-60:], inst=i, hand=hand, carry=cs, arrive=arr["t"], need=float(need),
                    full=float(full), att_orn=list(arr["att"].values())[0][1], dx=float(np.median(pts[:, 0])),
                    dy=float(np.median(pts[:, 1])), n=len(pts)))
    print(json.dumps(out[-1]), flush=True)
json.dump(out, open("/workspace/persistent/diag/dotcal/dotcal.json", "w"))
