"""Summarize a per-step gripper trace written by serve_indep.py (B1K_TRACE_DIR).

    python diag/gripper_trace_summary.py <trace_epN.csv> [bin_steps=500]

Values are normalized: -1 = fully closed, +1 = fully open. cmd = what the policy sent, meas = finger width from proprio.
Per bin: mean cmd / meas per hand, number of commanded open<->close flips, and the share of steps where the command
and the measured width disagree by > 0.5 (a lasting gap means the fingers do not follow the command; cmd closed with
meas stuck part-open usually means an object is between the fingers).
Ends with the list of commanded open/close events (cmd crossing 0) and how many steps the fingers took to follow.
"""
import csv
import sys

path = sys.argv[1]
binsz = int(sys.argv[2]) if len(sys.argv) > 2 else 500
rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(path))]
print(f"{path}: {len(rows)} steps")
print(f"{'steps':>12} {'stage':>5} | {'cmdL':>6} {'measL':>6} {'flipL':>5} {'gapL%':>5} | {'cmdR':>6} {'measR':>6} {'flipR':>5} {'gapR%':>5}")
for b0 in range(0, len(rows), binsz):
    seg = rows[b0:b0 + binsz]
    out = []
    for h in "LR":
        c = [r[f"cmd_{h}"] for r in seg]
        m = [r[f"meas_{h}"] for r in seg]
        flips = sum(1 for a, b in zip(c, c[1:]) if (a > 0) != (b > 0))
        gap = 100 * sum(1 for x, y in zip(c, m) if abs(x - y) > 0.5) / len(seg)
        out.append(f"{sum(c)/len(c):6.2f} {sum(m)/len(m):6.2f} {flips:5d} {gap:5.0f}")
    st = max(set(int(r["stage"]) for r in seg), key=[int(r["stage"]) for r in seg].count)
    print(f"{int(seg[0]['step']):>5}-{int(seg[-1]['step']):<6} {st:>5} | {out[0]} | {out[1]}")

print("\ncommanded open/close events (cmd crosses 0), fingers' response:")
for h in "LR":
    for i in range(1, len(rows)):
        a, b = rows[i - 1][f"cmd_{h}"], rows[i][f"cmd_{h}"]
        if (a > 0) == (b > 0):
            continue
        want_open = b > 0
        lag = None
        for j in range(i, min(i + 300, len(rows))):
            m = rows[j][f"meas_{h}"]
            if (want_open and m > 0.5) or (not want_open and m < -0.5):
                lag = j - i
                break
        end_meas = rows[min(i + 300, len(rows) - 1)][f"meas_{h}"]
        resp = f"followed in {lag} steps" if lag is not None else f"NOT followed within 300 steps (meas after 300 = {end_meas:.2f})"
        print(f"  {h} step {int(rows[i]['step']):>6}: {'OPEN ' if want_open else 'CLOSE'} -> {resp}")
