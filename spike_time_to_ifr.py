import numpy as np
from typing import List
from joblib import Parallel, delayed


def _spike_time_to_ifr(spike_time, time_res, first_time, last_time):
    ss_diff = np.diff((spike_time - first_time) * time_res, prepend=0., append=(last_time - first_time) * time_res)
    ss_diff = ss_diff[ss_diff != 0.]
    ifr_vals = time_res / ss_diff
    repeats = np.round(ss_diff).astype(int)
    return np.repeat(ifr_vals, repeats)


def spike_time_to_ifr(
        spike_times: List[np.ndarray],
        time_res: float,
        first_times=None,
        last_times=None,
        n_jobs: int = -1
) -> List[np.ndarray]:
    """
    Converts a list of spike times to IFR time intervals.

    :param spike_times: The list of spike times for each trial. For instance:
        [
            np.array([-0.2, -0.063, 0.045]),
            np.array([-0.125, -0.1, -0.08, 0., 0.015, 0.017, 0.019, 0.096, 0.15]),
        ]
    :param time_res: Time resolution for output IFR. Set to 1000 for having ms resolution.
    :param first_times: A list of first time points for each trial. If None, the first spike time is used.
    :param last_times: A list of last time points for each trial. If None, the last spike time is used.
    :param n_jobs: The number of jobs to run in parallel. Specify `-1` to use all available cores.
    :return: A list of IFR time intervals for each trial.
    """
    if first_times is None:
        first_times = [trl_spk_time[0] for trl_spk_time in spike_times]

    if last_times is None:
        last_times = [trl_spk_time[-1] for trl_spk_time in spike_times]

    return Parallel(n_jobs=n_jobs)(
        delayed(_spike_time_to_ifr)(st, time_res, ft, lt)
        for (st, ft, lt) in zip(spike_times, first_times, last_times)
    )
