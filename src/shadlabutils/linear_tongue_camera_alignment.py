import json

import numpy as np
import pandas as pd
from pathlib import Path, PureWindowsPath

from marmapp_funcs import read_compressed_values, EyeTrackerData
from shadlabutils.alignment_utils import binary_agreement, fit_clock_mapping
from shadlabutils.closest_times import closest_times

score_threshold = 0.85


def linear_tongue_camera_alignment(path_to_raw: str, camera_label: str = "side"):
    path_to_raw = Path(PureWindowsPath(path_to_raw))

    # read settings
    event_jsonl_file = [i for i in path_to_raw.glob("*.jsonl") if
                        "diagnostics" not in str(i) and "spinnaker" not in str(i)]
    assert len(event_jsonl_file) == 1, f"There should be only one event file, but there are {len(event_jsonl_file)}"
    event_jsonl_file = event_jsonl_file[0]

    settings = None
    with open(event_jsonl_file, "r") as f:
        for l in f:
            d = json.loads(l)
            if d["type"] == "VpixxEndEvent":
                continue
            if d["kind"] == "UPDATE_SETTINGS":
                settings = d["data"].copy()
            if d["kind"] == "UPDATE_PARAMETERS" and "exp_beginning_perf_time" in d["data"]:
                exp_beginning_perf_time = d["data"]["exp_beginning_perf_time"]

    # read eye tracker din to extract the reference signal
    eye_tracker_file = [i for i in path_to_raw.glob("*_eye_data.bin")]
    assert len(eye_tracker_file) == 1, f"There should be only one eye data file, but there are {len(eye_tracker_file)}"
    eye_data_list = read_compressed_values(eye_tracker_file[0])
    eye_tracker_data_list = [
        EyeTrackerData(*eye_data_list[i:i + 19])
        for i in range(0, len(eye_data_list), 19)
        if len(eye_data_list[i:i + 19]) == 19
    ]
    time_received = np.array([d.time_received for d in eye_tracker_data_list])
    tongue_ref = np.array([(e.din >> int(settings["din_tongue_ttl"])) & 1 for e in eye_tracker_data_list])

    # read ttl events from tongue camera
    ttl_csv = [i for i in path_to_raw.glob("*.csv") if camera_label in str(i) and "metadata" in str(i)]

    assert len(ttl_csv) == 1, (f"There should be only one metadata file, but there are "
                               f"{len(ttl_csv)} for {camera_label} camera")

    ttl = pd.read_csv(ttl_csv[0])

    # Now do the alignment (side camera)
    tref_ = time_received[np.diff(tongue_ref, prepend=0) == 1]
    xref_ = np.ones_like(tref_)

    between_ = tref_ + 1e-3  # (tref_[:-1] + tref_[1:]) / 2
    zeros_ = np.zeros_like(between_)
    tref = np.zeros((tref_.shape[0] + between_.shape[0],))
    tref[::2] = tref_
    tref[1::2] = between_
    xref = np.zeros((xref_.shape[0] + zeros_.shape[0],))
    xref[::2] = xref_
    xref[1::2] = zeros_

    tsig_ = ttl["camera_timestamp"].values * 1e-9
    xsig_ = np.ones_like(tsig_)

    between_ = tsig_ + 1e-3  # (tsig_[:-1] + tsig_[1:]) / 2
    zeros_ = np.zeros_like(between_)
    tsig = np.zeros((tsig_.shape[0] + between_.shape[0],))
    tsig[::2] = tsig_
    tsig[1::2] = between_
    xsig = np.zeros((xsig_.shape[0] + zeros_.shape[0],))
    xsig[::2] = xsig_
    xsig[1::2] = zeros_

    first_tref_time = tref[tref >= exp_beginning_perf_time][0]
    first_tsig_time = tsig[0]

    success = False
    for i_try in range(5):
        a, b, score = fit_clock_mapping(tref - first_tref_time, xref, tsig - first_tsig_time, xsig,
                                        drift_ppm=5000, offset_range=(-1, 1))
        tsig_aligned = a * (tsig - first_tsig_time) + b + first_tref_time

        diff_sig_0 = closest_times(tsig_aligned[xsig == 0], tref[xref == 0]) - tref[xref == 0]
        diff_sig_1 = closest_times(tsig_aligned[xsig == 1], tref[xref == 1]) - tref[xref == 1]
        time_diff = np.zeros_like(tref)
        time_diff[xref == 0] = diff_sig_0
        time_diff[xref == 1] = diff_sig_1

        a_corrected, b_corrected = np.polyfit(tref[np.abs(time_diff) < 0.01] - first_tref_time,
                                              time_diff[np.abs(time_diff) < 0.01], deg=1)
        err = time_diff - (a_corrected * (tref - first_tref_time) + b_corrected)
        a_corrected, b_corrected = np.polyfit(tref[np.abs(err) < 0.002] - first_tref_time,
                                              time_diff[np.abs(err) < 0.002], deg=1)
        c, d = np.histogram(err[np.abs(err) < 0.01], 100)
        b_corrected += d[np.argmax(c)]
        a -= a_corrected
        b -= b_corrected

        score = -binary_agreement((a, b), tref - first_tref_time, xref, tsig - first_tsig_time, xsig)

        tsig_aligned = a * (tsig - first_tsig_time) + b + first_tref_time
        diff_sig_0 = closest_times(tsig_aligned[xsig == 0], tref[xref == 0]) - tref[xref == 0]
        diff_sig_1 = closest_times(tsig_aligned[xsig == 1], tref[xref == 1]) - tref[xref == 1]
        time_diff = np.zeros_like(tref)
        time_diff[xref == 0] = diff_sig_0
        time_diff[xref == 1] = diff_sig_1

        if score > score_threshold:
            success = True
            break

    assert success, f"Alignment failed after 5 tries. Binary agreement score: {score}"
    return a, - a * first_tsig_time + b + first_tref_time - exp_beginning_perf_time, score, time_diff
