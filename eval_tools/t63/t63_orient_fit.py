"""Fit the constant goal rotation G (nail-frame relative orientation of the detector when attached) from pose logs:
att_deg(t) = angle(G^-1 * Qrel(t)), Qrel = q_nail^-1 * q_alarm (object frames, xyzw). Then decompose the remaining error
at each record into twist about the wall normal vs tilt. usage: t63_orient_fit.py pose.jsonl [...]"""
import json, sys, numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
recs = []
for f in sys.argv[1:]:
    for l in open(f):
        r = json.loads(l)
        if "att" in r and "objq" in r:
            a = list(r["att"].values())[0]
            qn, qa = r["objq"]["wall_nail.n.01_1"], r["objq"]["fire_alarm.n.02_1"]
            recs.append((f, r["t"], R.from_quat(qn), R.from_quat(qa), a[0], a[1], r["base"][2]))
print(len(recs), "records")
Q = [rn.inv() * ra for _, _, rn, ra, *_ in recs]
ang = np.array([x[5] for x in recs])
def res(v):
    G = R.from_rotvec(v)
    return np.array([np.degrees((G.inv() * q).magnitude()) for q in Q]) - ang
best = min((least_squares(res, np.random.default_rng(s).normal(size=3) * 2) for s in range(30)), key=lambda o: o.cost)
G = R.from_rotvec(best.x)
print("fit rms deg", np.sqrt(np.mean(best.fun ** 2)).round(2), "G rotvec", best.x.round(3))
# wall normal in nail frame: horizontal direction base heading at the closest records ~ into the wall
for (f, t, rn, ra, d, a, yaw), q in zip(recs, Q):
    if d > 0.25:
        continue
    E = q * G.inv()  # error rotation in nail frame (apply to goal to get current)... express in world:
    Ew = rn * E * rn.inv()
    n = np.array([np.cos(yaw), np.sin(yaw), 0.0])  # robot forward = wall normal (robot faces the wall)
    rv = Ew.as_rotvec(); tw = np.dot(rv, n)
    twist = R.from_rotvec(n * tw)  # approx swing-twist via projection of the axis
    # exact swing-twist decomposition
    qx = Ew.as_quat(); p = np.dot(qx[:3], n) * n; tq = np.r_[p, qx[3]]; tq /= np.linalg.norm(tq)
    T_ = R.from_quat(tq); S = Ew * T_.inv()
    tdeg = np.degrees(T_.magnitude()) * np.sign(np.dot(T_.as_rotvec(), n) or 1)
    print(f"{f.split('/')[-3][-22:]:>22} {f.split('_')[-1][:-6]:>4} t={t:5d} dist={d:.3f} att={a:6.1f} "
          f"fit={np.degrees(E.magnitude()):6.1f}  twist(about wall normal)={tdeg:+7.1f}  tilt={np.degrees(S.magnitude()):5.1f}")
