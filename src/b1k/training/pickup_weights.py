"""Per-frame sampling weights around pickups (task 63 grasp emphasis)."""
import numpy as np

GRIPPER_DIMS = (14, 22)  # left, right; absolute commands, -1 closed / +1 open
CARRY_MIN_LEN = 200       # a close held this long is a carrying grasp


def _segments(closed):
    x = np.diff(np.concatenate([[0], closed.astype(np.int8), [0]]))
    return list(zip(np.where(x == 1)[0], np.where(x == -1)[0]))


def grasp_weights(actions, episode_index, *, approach=(150, 10), approach_weight=4.0, close_weight=1.0,
                  lift=60, lift_weight=3.0, recovery_weight=8.0):
    """actions [N, 23], episode_index [N] (rows of each episode contiguous, in frame order).

    For every carrying grasp (close held >= CARRY_MIN_LEN frames, either hand) at frame c:
    frames [c-150, c-10) get approach_weight, [c-10, c] close_weight, (c, c+60] lift_weight.
    Episodes with a short close (< CARRY_MIN_LEN) before their first carrying grasp are retries:
    frames from 150 before that short close to 60 after the carrying close get recovery_weight.
    """
    w = np.ones(len(actions), dtype=np.float64)
    recovery = np.zeros(len(actions), dtype=bool)
    starts = np.flatnonzero(np.r_[True, episode_index[1:] != episode_index[:-1]])
    ends = np.r_[starts[1:], len(actions)]
    n_carry = n_retry_eps = 0
    for s, e in zip(starts, ends):
        closes = sorted((a, b, b - a >= CARRY_MIN_LEN) for d in GRIPPER_DIMS for a, b in _segments(actions[s:e, d] < 0))
        carries = [a for a, b, long in closes if long]
        for c in carries:
            n_carry += 1
            lo = max(c - approach[0], 0)
            w[s + lo:s + max(c - approach[1], 0)] = approach_weight
            w[s + max(c - approach[1], 0):s + c + 1] = close_weight
            w[s + c + 1:s + min(c + lift + 1, e - s)] = lift_weight
        if carries:
            short = [a for a, b, long in closes if not long and a < carries[0]]
            if short:
                n_retry_eps += 1
                sl = slice(s + max(short[0] - approach[0], 0), s + min(carries[0] + lift + 1, e - s))
                w[sl] = np.maximum(w[sl], recovery_weight)
                recovery[sl] = True
    return w, dict(episodes=len(starts), carrying_grasps=n_carry, retry_episodes=n_retry_eps, recovery=recovery)
