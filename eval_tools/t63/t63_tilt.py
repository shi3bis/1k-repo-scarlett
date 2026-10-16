"""Split the detector->nail attach error into roll (about wall normal), yaw (about vertical) and pitch (about the
along-wall axis), wall normal = horizontal nail axis. Uses G fitted by t63_orient_fit.py. usage: t63_tilt.py pose.jsonl..."""
import json, sys, numpy as np
from scipy.spatial.transform import Rotation as R
G = R.from_rotvec([0.698, 9.397, -0.041])
for f in sys.argv[1:]:
    for l in open(f):
        r = json.loads(l)
        if "att" not in r or "objq" not in r:
            continue
        d, a = list(r["att"].values())[0][:2]
        if d > 0.25:
            continue
        rn, ra = R.from_quat(r["objq"]["wall_nail.n.01_1"]), R.from_quat(r["objq"]["fire_alarm.n.02_1"])
        goal = rn * G                       # alarm world orientation when attached
        Ew = ra * goal.inv()                # world-frame error rotation
        ax = rn.as_matrix()                 # nail axes (columns)
        hz = [i for i in range(3) if abs(ax[2, i]) < 0.3]
        yaw = r["base"][2]; h = np.array([np.cos(yaw), np.sin(yaw), 0])
        i = max(hz, key=lambda i: abs(ax[:, i] @ h)); n = ax[:, i] * np.sign(ax[:, i] @ h); n[2] = 0; n /= np.linalg.norm(n)
        z = np.array([0, 0, 1.0]); s = np.cross(z, n)  # along-wall
        M = np.c_[n, s, z]
        roll, pitch, yw = (M.T @ Ew.as_matrix() @ M)  # error in (normal, along, up) frame
        e = R.from_matrix(M.T @ Ew.as_matrix() @ M).as_euler("xyz", degrees=True)
        print(f"{f.split('/')[-3][-14:]:>14} {f.split('_')[-1][:-6]:>4} t={r['t']:5d} dist={d:.3f} att={a:6.1f} "
              f"roll(wall normal)={e[0]:+7.1f} pitch(nose up/down)={e[1]:+7.1f} yaw(left/right)={e[2]:+7.1f}  "
              f"base-vs-wall {np.degrees(np.arctan2(np.cross(h, n)[2], h @ n)):+5.1f}")
