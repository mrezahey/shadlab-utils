import numpy as np
from numba import njit

from shadlabutils.quicksort import quicksort


@njit
def histdiff(data1, data2, bins):
    """
    Similar to "Other Packages/Spikes/analysis/helpers/histdiff.c" but implemented in numba.

    :param data1: 1D numpy array of spike times 1 with arbitrary length.
    :param data2: 1D numpy array of spike times 2 with arbitrary length.
    :param bins: 1D numpy array of bins with size of b
    :return: counts: 1D counts with size of b-1
             centers: 1D center of bins with size of b-1
    """
    # data1 = quicksort(data1)
    # data2 = quicksort(data2)

    n1 = len(data1)
    n2 = len(data2)
    nbins = len(bins) - 1

    counts = np.zeros(nbins, dtype=np.int64)
    centers = 0.5 * (bins[:-1] + bins[1:])

    j_start = 0

    for i in range(n1):
        t1 = data1[i]

        # advance j_start to skip data2 points too early
        while j_start < n2 and data2[j_start] < t1 + bins[0]:
            j_start += 1

        j = j_start

        while j < n2:
            diff = data2[j] - t1
            if diff >= bins[-1]:
                break

            # bin the difference
            for b in range(nbins):
                if bins[b] <= diff < bins[b + 1]:
                    counts[b] += 1
                    break

            j += 1

    return counts, centers
