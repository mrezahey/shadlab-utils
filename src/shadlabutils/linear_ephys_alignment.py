import json
import os

import numpy as np
import pandas as pd
from pathlib import Path, PureWindowsPath

from shadlabutils.alignment_utils import binary_agreement, fit_clock_mapping
from shadlabutils.closest_times import closest_times

sampling_freq = 30000
score_threshold = 0.85


def linear_ephys_alignment(path_to_raw: str):
    path_to_raw = Path(PureWindowsPath(path_to_raw))

    # read events to extract the reference signal
    event_jsonl_file = [i for i in path_to_raw.glob("*.jsonl") if "diagnostics" not in str(i)]
    assert len(event_jsonl_file) == 1, f"There should be only one event file, but there are {len(event_jsonl_file)}"
    event_jsonl_file = event_jsonl_file[0]

    random_sig_ref = list()
    last_ts = 0
    settings = None
    with open(event_jsonl_file, "r") as f:
        for l in f:
            d = json.loads(l)
            if d["type"] == "VpixxEndEvent":
                continue
            last_ts = d["ts"]
            if d["kind"] == "UPDATE_SETTINGS":
                settings = d["data"].copy()
            if d["kind"] == "UPDATE_PARAMETERS" and "exp_beginning_perf_time" in d["data"]:
                exp_beginning_perf_time = d["data"]["exp_beginning_perf_time"]
            if d["kind"] == "SET_DOUT" and settings is not None and d["payload"]["pin_num"] == settings["dout_random_sig"]:
                random_sig_ref.append([d["payload"]["value"], d["ts"]])

    random_sig_ref = pd.DataFrame(random_sig_ref, columns=["value", "time"])

    # read ttl events from ephys
    openephys = [i for i in path_to_raw.glob("*all_channels.events")]
    spikeglx = [i for i in path_to_raw.glob("*.ap.bin")]
    intanrhs = [i for i in path_to_raw.glob("*.rhs")]

    assert len(openephys) + len(spikeglx) + len(intanrhs) == 1, (f"There should be only one ephys file, "
                                                                 f"but there are {len(openephys)} openephys, "
                                                                 f"{len(spikeglx)} spikeglx, {len(intanrhs)} intanrhs")
    ephys_file = openephys[0] if len(openephys) == 1 else spikeglx[0] if len(spikeglx) == 1 else intanrhs[0]
    ephys_type = "OpenEphys" if len(openephys) == 1 else "SpikeGLX" if len(spikeglx) == 1 else "IntanRHX"

    if ephys_type == "IntanRHX":
        from shadlabutils.intanutil.data import calculate_data_size  # , check_end_of_file
        from shadlabutils.intanutil.header import read_header
        from shadlabutils.intanutil.intan_read_ttl_fast import read_all_data_blocks

        random_sig_channel = 0
        with open(ephys_file, 'rb') as fid:
            header = read_header(fid)
            data_present, filesize, num_blocks, num_samples = calculate_data_size(header, ephys_file, fid)
            if data_present:
                data = read_all_data_blocks(header, num_samples, num_blocks, fid)
                # check_end_of_file(filesize, fid)

        vals_ = data["board_dig_in_raw"] >> random_sig_channel & 1
        where_changed = np.diff(vals_, prepend=-1) != 0

        ttl = pd.DataFrame()
        ttl["timestamps"] = data["t"][where_changed] / sampling_freq
        ttl["eventId"] = vals_[where_changed]



    elif ephys_type == "SpikeGLX":
        ephys_rise_file = [i for i in path_to_raw.glob("*.imec0.ap.xd_*_0.txt")][0]
        ephys_fall_file = [i for i in path_to_raw.glob("*.imec0.ap.xid_*_0.txt")][0]
        rising_times = np.loadtxt(ephys_rise_file)
        falling_times = np.loadtxt(ephys_fall_file)

        ttl = pd.concat([
            pd.DataFrame({"timestamps": rising_times, "eventId": np.ones_like(rising_times)}),
            pd.DataFrame({"timestamps": falling_times, "eventId": np.zeros_like(falling_times)})
        ], ignore_index=True).sort_values("timestamps").reset_index(drop=True)

        meta = {}
        with open([i for i in path_to_raw.glob("*.imec0.ap.meta")][0], "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    meta[key.lstrip("~")] = value


        fs = float(meta["imSampRate"])
        n_channels = int(meta["nSavedChans"])
        first_sample = int(meta["firstSample"])
        file_size = os.path.getsize([i for i in path_to_raw.glob("*.imec0.ap.bin")][0])
        n_samples = file_size // (2 * n_channels)


    elif ephys_type == "OpenEphys":
        random_sig_channel = 1
        event_dtype = np.dtype([
            ("timestamps", "<i8"),
            ("sampleNum", "<i2"),
            ("eventType", "<u1"),
            ("nodeId", "<u1"),
            ("eventId", "<u1"),
            ("channel", "<u1"),
            ("recordingNumber", "<u2"),
        ])

        with open(ephys_file, "rb") as f:
            _ = f.read(1024)
            events_array = np.fromfile(f, dtype=event_dtype)

        events = pd.DataFrame({name: events_array[name] for name in events_array.dtype.names})
        events["timestamps"] = events["timestamps"] / sampling_freq
        ttl = events[(events["eventType"] == 3) & (events["channel"] == random_sig_channel)]
        ttl = ttl.loc[:, ["timestamps", "eventId"]]

        time_file = sorted([i for i in path_to_raw.glob("*.continuous")])
        assert len(time_file) >= 1, f"There should be at least one continuous file"
        time_file = time_file[0]

    # Now do the alignment
    tref = random_sig_ref["time"].values
    xref = random_sig_ref["value"].values
    tsig = ttl["timestamps"].values
    xsig = ttl["eventId"].values

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
