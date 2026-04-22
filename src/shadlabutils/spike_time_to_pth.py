import numpy as np
from typing import List, Optional
from scipy.signal import lfilter
from joblib import Parallel, delayed


def _compute_trial_histogram(trl_spk_time, edges, time_span):
    count, _ = np.histogram(trl_spk_time, edges)
    return np.interp(time_span, edges[1:], count)


def spike_time_to_pth(
        spike_times: List[np.ndarray],
        time_span: np.ndarray,
        binwidth: int = 5,
        mult_factor: float = 1000.,
        smooth_filter: Optional[float] = 2,
        n_jobs: int = -1
) -> np.ndarray:
    """
    Convert spike times to pth by counting the spikes in each bin.
    An optional causal moving average filter is applied to the counts.

    :param spike_times: The list of spike times for each trial. For instance:
        [
            np.array([-200, -63, 45]),
            np.array([-125, -100, -80, 0, 15, 17, 19, 96, 150]),
        ]
    :param time_span: The time span that you want to compute the pth. You are responsible to
        make sure that the scale of this vector equals to the scale of `spike_times`. For instance:
        np.arange(-250, 251)
    :param binwidth: The bin width of the pth. This parameter has no scale.
    :param mult_factor: The multiplication factor used to scale the output pth. If the timings are in ms
        for instance, this value should be 1000.
    :param smooth_filter: The smoothing filter window length. If `None`, no smoothing is applied.
    :param n_jobs: The number of jobs to run in parallel. Specify `-1` to use all available cores.
    :return: The pth of spikes with the shape of (number of trials x number of time points).
    """
    if smooth_filter is not None:
        smooth_filter = int(smooth_filter * binwidth)

    time_span_extend = np.concatenate([
        time_span,
        np.arange(time_span[-1] + 1, time_span[-1] + 2 * binwidth + 1)
    ])
    edges = time_span_extend[::binwidth] - binwidth / 2

    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_trial_histogram)(trl_spk_time, edges, time_span)
        for trl_spk_time in spike_times
    )

    out = np.vstack(results) * mult_factor / binwidth
    if smooth_filter is not None:
        out = lfilter([1 / smooth_filter] * smooth_filter, [1.], out)

    return out
