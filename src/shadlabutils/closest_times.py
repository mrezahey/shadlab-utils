import numpy as np


def closest_times(ref, sig, return_idx=False):
    idx_right = np.searchsorted(ref, sig)

    idx_left = np.clip(idx_right - 1, 0, len(ref) - 1)
    idx_right = np.clip(idx_right, 0, len(ref) - 1)

    choose_right = np.abs(ref[idx_right] - sig) < np.abs(ref[idx_left] - sig)

    idx_closest = np.where(choose_right, idx_right, idx_left)
    if return_idx:
        return ref[idx_closest], idx_closest
    else:
        return ref[idx_closest]
