import ast
import os
from typing import List, Optional, Tuple

from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.io import loadmat

from pyutils.extract_cell_metadata import extract_cell_metadata
from pyutils.load_cell import load_cell
from pyutils.load_eye_traces import load_eye_traces
from pyutils.load_tongue_traces import load_tongue_traces


def _load_one_session(
    path_ephys: str,
    this_animal: str,
    this_session: str,
    unit_type: List[str],
) -> Tuple[List[list], List[list]]:
    recs = []
    units = []

    path_data_monkey_sorted_ = os.path.join(path_ephys, f"data_{this_animal}")

    metadata = extract_cell_metadata(path_data_monkey_sorted_, this_session)
    list_units_this_sess = sorted([
        metadata["cell_list"][ind] for ind, t in enumerate(metadata["cell_type"]) if t in unit_type
    ])

    if len(list_units_this_sess) == 0:
        return recs, units

    session_unit_presence = []
    ss_times_per_unit = []
    actual_ss_times_per_unit = []
    cs_times_per_unit = []
    actual_cs_times_per_unit = []

    for this_unit_file_name in list_units_this_sess:
        data = load_cell(path_data_monkey_sorted_, this_unit_file_name[:6], this_unit_file_name)
        session_unit_presence.append(data["rec_info"]["rec_flag"])

        this_ss = data["data_recordings"]["Neural_Data"]
        this_ss = [s["SS_time"] for s in this_ss] if isinstance(this_ss, list) else [this_ss["SS_time"]]
        this_cs = data["data_recordings"]["Neural_Data"]
        this_cs = [s["CS_time"] for s in this_cs] if isinstance(this_cs, list) else [this_cs["CS_time"]]

        this_ss_prop = data["data_recordings"]["Neural_Prop"]
        this_ss_prop = [s["SS_time"] for s in this_ss_prop] if isinstance(this_ss_prop, list) else [this_ss_prop["SS_time"]]
        this_cs_prop = data["data_recordings"]["Neural_Prop"]
        this_cs_prop = [s["CS_time"] for s in this_cs_prop] if isinstance(this_cs_prop, list) else [this_cs_prop["CS_time"]]

        ss_times_per_unit.append(this_ss)
        actual_ss_times_per_unit.append(this_ss_prop)
        cs_times_per_unit.append(this_cs)
        actual_cs_times_per_unit.append(this_cs_prop)

    session_unit_presence = np.stack(session_unit_presence)

    # Load eye traces for the first unit's recording date
    this_unit_file_name = list_units_this_sess[0]
    eye_traces = load_eye_traces(path_data_monkey_sorted_, this_unit_file_name[:6])

    len_recs_before = len(recs)

    # Recs + Units assembly
    for ind_rec, rec in enumerate(data["rec_info"]["rec_list"]):  # NoQA
        recs.append([
            this_animal,
            this_session,
            rec[0],
            eye_traces["time"][ind_rec],
            eye_traces["X"][ind_rec],
            eye_traces["Y"][ind_rec],
            eye_traces["fix"][ind_rec]["ind_onset"].astype(int),
            eye_traces["fix"][ind_rec]["ind_offset"].astype(int),
            eye_traces["sac"][ind_rec]["tag"].astype(int),
            eye_traces["sac"][ind_rec]["ind_onset"].astype(int),
            eye_traces["sac"][ind_rec]["ind_offset"].astype(int),
            eye_traces["sac"][ind_rec]["ind_vmax"].astype(int),
            eye_traces["sac"][ind_rec]["ind_visual"].astype(int),
        ])

        for this_unit_ind in np.where(session_unit_presence[:, ind_rec])[0]:
            s1_ = ss_times_per_unit[this_unit_ind].pop(0)
            s2_ = actual_ss_times_per_unit[this_unit_ind].pop(0)
            s3_ = cs_times_per_unit[this_unit_ind].pop(0)
            s4_ = actual_cs_times_per_unit[this_unit_ind].pop(0)
            units.append([
                list_units_this_sess[this_unit_ind][:-4],
                [metadata["cell_type"][ind] for ind, t in enumerate(metadata["cell_list"]) if t == list_units_this_sess[this_unit_ind]][0],
                ind_rec + len_recs_before,
                s1_ if s1_ is not None else np.array([]),
                s2_ if s2_ is not None else np.array([]),
                s3_ if s3_ is not None else np.array([]),
                s4_ if s4_ is not None else np.array([]),
            ])

    return recs, units

def load_session(
        path_ephys: str,
        animals: List[str],
        sessions: List[str],
        unit_type: List[str],
        load_left_right: bool = True,
        path_cal_mat: Optional[str] = None,
        load_tongue: bool = True,
        is_rpe_task: bool = False,
) -> (pd.DataFrame, pd.DataFrame):

    recs = []
    units = []

    with ProcessPoolExecutor() as executor:
        results = executor.map(
            _load_one_session,
            [path_ephys] * len(animals),
            animals,
            sessions,
            [unit_type] * len(animals),
        )

        c = 0
        for r, u in results:
            for i in range(len(u)):
                u[i][2] += c
            c += len(r)
            recs.extend(r)
            units.extend(u)

    recs = pd.DataFrame(
        recs,
        columns=[
            "animal", "session", "rec", "time", "X", "Y", "fix_ind_onset", "fix_ind_offset",
            "sac_tag", "sac_ind_onset", "sac_ind_offset", "sac_ind_vmax", "sac_ind_visual"
        ]
    )

    units = pd.DataFrame(
        units,
        columns=["name", "cell_type", "recs_ind", "ss_time", "actual_ss_time", "cs_time", "actual_cs_time"]
    )

    units["ss_first_time"] = units["ss_time"].apply(lambda v: v[0] if len(v) > 0 else np.inf)
    units.loc[np.isinf(units["ss_first_time"].values), "ss_first_time"] = units.loc[np.isinf(units["ss_first_time"].values), "cs_time"]\
        .apply(lambda v: v[0] if len(v) > 0 else np.inf)
    units["ss_first_time"] = units[["recs_ind", "ss_first_time"]].groupby("recs_ind")["ss_first_time"].transform("min")
    units["ss_last_time"] = units["ss_time"].apply(lambda v: v[-1] if len(v) > 0 else -np.inf)
    units.loc[np.isinf(units["ss_last_time"].values), "ss_last_time"] = units.loc[np.isinf(units["ss_last_time"].values), "cs_time"]\
        .apply(lambda v: v[-1] if len(v) > 0 else -np.inf)
    units["ss_last_time"] = units[["recs_ind", "ss_last_time"]].groupby("recs_ind")["ss_last_time"].transform("max") + 1

    recs["sac_eye_ang"] = [np.array([]) for _ in range(len(recs))]
    recs["sac_vis_ang"] = [np.array([]) for _ in range(len(recs))]
    recs["sac_vis_amp"] = [np.array([]) for _ in range(len(recs))]

    if is_rpe_task:
        recs["task_cond"] = [np.array([]) for _ in range(len(recs))]
        recs["tgt_cond"] = [np.array([]) for _ in range(len(recs))]
        recs["jump_cond"] = [np.array([]) for _ in range(len(recs))]
        recs["rew_cond"] = [np.array([]) for _ in range(len(recs))]
        recs["choice"] = [np.array([]) for _ in range(len(recs))]
        recs["cue_x_high_rew"] = [np.array([]) for _ in range(len(recs))]
        recs["cue_x_low_rew"] = [np.array([]) for _ in range(len(recs))]
        recs["cue_y_high_rew"] = [np.array([]) for _ in range(len(recs))]
        recs["cue_y_low_rew"] = [np.array([]) for _ in range(len(recs))]

    for this_unit_ind, this_unit in units.iterrows():
        if recs.loc[this_unit["recs_ind"], "sac_eye_ang"].shape[0] > 0:
            continue

        path_data_monkey_sorted_ = os.path.join(path_ephys, f"data_{recs.loc[this_unit['recs_ind'], 'animal']}")
        this_unit_file_name = f'{this_unit["name"]}.mat'
        data = load_cell(path_data_monkey_sorted_, this_unit_file_name[:6], this_unit_file_name)

        if type(data["data_recordings"]["eye"]) == list:
            this_valid_recs = [
                data["rec_info"]["rec_list"][i][0] for i, id_ in enumerate(data["rec_info"]["rec_flag"]) if id_
            ]
            idx = [id_ for id_, i in enumerate(this_valid_recs) if i == recs.loc[this_unit["recs_ind"], "rec"]][0]
            info = data["data_recordings"]["eye"][idx]
        else:
            info = data["data_recordings"]["eye"]

        recs.at[this_unit["recs_ind"], "sac_eye_ang"] = info["sac"]["eye_ang"]
        recs.at[this_unit["recs_ind"], "sac_vis_ang"] = info["sac"]["vis_ang"]
        recs.at[this_unit["recs_ind"], "sac_vis_amp"] = info["sac"]["vis_amp"]

        if is_rpe_task:
            recs.at[this_unit["recs_ind"], "task_cond"] = info["sac"]["task_cond"]
            recs.at[this_unit["recs_ind"], "tgt_cond"] = info["sac"]["tgt_cond"]
            recs.at[this_unit["recs_ind"], "jump_cond"] = info["sac"]["jump_cond"]
            recs.at[this_unit["recs_ind"], "rew_cond"] = info["sac"]["rew_cond"]
            recs.at[this_unit["recs_ind"], "choice"] = info["sac"]["choice"]
            recs.at[this_unit["recs_ind"], "cue_x_high_rew"] = info["sac"]["cue_x_high_rew"]
            recs.at[this_unit["recs_ind"], "cue_x_low_rew"] = info["sac"]["cue_x_low_rew"]
            recs.at[this_unit["recs_ind"], "cue_y_high_rew"] = info["sac"]["cue_y_high_rew"]
            recs.at[this_unit["recs_ind"], "cue_y_low_rew"] = info["sac"]["cue_y_low_rew"]


    if load_left_right:
        cal_mat = pd.read_csv(path_cal_mat)
        cal_mat['cal_mat_l'] = cal_mat['cal_mat_l'].apply(lambda x: np.array(ast.literal_eval(x)))
        cal_mat['cal_mat_r'] = cal_mat['cal_mat_r'].apply(lambda x: np.array(ast.literal_eval(x)))

        recs["XL"] = [np.array([]) for _ in range(len(recs))]
        recs["YL"] = [np.array([]) for _ in range(len(recs))]
        recs["XR"] = [np.array([]) for _ in range(len(recs))]
        recs["YR"] = [np.array([]) for _ in range(len(recs))]
        recs["blink"] = [np.array([]) for _ in range(len(recs))]

        for this_rec_ind, this_rec in recs.iterrows():
            cm_ = cal_mat.loc[
                (cal_mat["animal"] == this_rec["animal"]) & (cal_mat["session"] == this_rec["session"])
                ].iloc[0]

            path_data_monkey_sorted = os.path.join(path_ephys, f"data_{this_rec['animal']}")
            rec = this_rec["rec"]
            raw_dir = str(os.path.join(path_data_monkey_sorted, rec[:7], rec[:10], rec, "raw_data"))

            data = loadmat(str(os.path.join(raw_dir,
                                            [f for f in os.listdir(raw_dir) if
                                             ("corrective_saccade" in f or "reward_prediction" in f) and ".mat" in f][
                                                0], )))
            data = {
                field_name: data["data"][0, 0][ind]
                for ind, field_name in enumerate(data["data"][0, 0].dtype.names)
            }
            for key in data.keys():
                if not key.startswith("trial"):
                    continue
                data[key] = {
                    field_name: data[key][0, 0][ind]
                    for ind, field_name in enumerate(data[key][0, 0].dtype.names)
                }

            if "device_time_data" in data["trial_1"]:
                t_ = list()
                lx_ = list()
                ly_ = list()
                rx_ = list()
                ry_ = list()
                b_ = list()

                for trl_ind in sorted([int(key.replace("trial_", "")) for key in data.keys() if key.startswith("trial")]):
                    t_.append(np.squeeze(data[f"trial_{trl_ind}"]["device_time_data"]))
                    lx_.append(np.squeeze(data[f"trial_{trl_ind}"]["eye_lx_raw_data"]))
                    ly_.append(np.squeeze(data[f"trial_{trl_ind}"]["eye_ly_raw_data"]))
                    rx_.append(np.squeeze(data[f"trial_{trl_ind}"]["eye_rx_raw_data"]))
                    ry_.append(np.squeeze(data[f"trial_{trl_ind}"]["eye_ry_raw_data"]))
                    b_.append(np.squeeze(
                        (data[f"trial_{trl_ind}"]["eye_l_blink_data"] + data[f"trial_{trl_ind}"]["eye_r_blink_data"]) > 0
                    ))

                t_ = np.concatenate(t_)
                lx_ = np.concatenate(lx_)
                ly_ = np.concatenate(ly_)
                rx_ = np.concatenate(rx_)
                ry_ = np.concatenate(ry_)
                b_ = np.concatenate(b_)

            else:
                t_ = np.squeeze(data["device_time_data"]).astype(float) / 1000.
                lx_ = np.squeeze(data["eye_lx_raw_data"])
                ly_ = np.squeeze(data["eye_ly_raw_data"])
                rx_ = np.squeeze(data["eye_rx_raw_data"])
                ry_ = np.squeeze(data["eye_ry_raw_data"])
                analyzed_eye_dir = str(
                    os.path.join(path_data_monkey_sorted, rec[:7], rec[:10], rec, "analyzed_data", "behavior_data", "eye")
                )
                pupil_data = loadmat(
                    str(os.path.join(analyzed_eye_dir,
                                     [f for f in os.listdir(analyzed_eye_dir) if "_PUPIL.mat" in f][0], ))
                )
                pupil_data = {
                    field_name: pupil_data["data"][0, 0][ind]
                    for ind, field_name in enumerate(pupil_data["data"][0, 0].dtype.names)
                }
                b_ = np.squeeze(np.isnan(pupil_data["pupil_data_r"]) | np.isnan(pupil_data["pupil_data_l"]))

            l_ = np.matmul(np.vstack([lx_, ly_, np.ones_like(lx_)]).T, cm_["cal_mat_l"])
            lx, ly = l_[:, 0], l_[:, 1]
            r_ = np.matmul(np.vstack([rx_, ry_, np.ones_like(rx_)]).T, cm_["cal_mat_r"])
            rx, ry = r_[:, 0], r_[:, 1]

            lx_interp = np.interp(this_rec["time"], t_, lx)
            ly_interp = np.interp(this_rec["time"], t_, ly)
            rx_interp = np.interp(this_rec["time"], t_, rx)
            ry_interp = np.interp(this_rec["time"], t_, ry)
            indices = np.searchsorted(t_, this_rec["time"], side="right") - 1
            b_interp = b_[indices]

            recs.at[this_rec_ind, "XL"] = lx_interp
            recs.at[this_rec_ind, "YL"] = ly_interp
            recs.at[this_rec_ind, "XR"] = rx_interp
            recs.at[this_rec_ind, "YR"] = ry_interp
            recs.at[this_rec_ind, "blink"] = b_interp

    if load_tongue:
        recs["tongue_time"] = [np.array([]) for _ in range(len(recs))]
        recs["tongue_tip_x"] = [np.array([]) for _ in range(len(recs))]
        recs["tongue_tip_y"] = [np.array([]) for _ in range(len(recs))]
        recs["tongue_l_x"] = [np.array([]) for _ in range(len(recs))]
        recs["tongue_l_y"] = [np.array([]) for _ in range(len(recs))]
        recs["tongue_r_x"] = [np.array([]) for _ in range(len(recs))]
        recs["tongue_r_y"] = [np.array([]) for _ in range(len(recs))]
        recs["tongue_mid_x"] = [np.array([]) for _ in range(len(recs))]
        recs["tongue_mid_y"] = [np.array([]) for _ in range(len(recs))]
        recs["lick_tag"] = [np.array([]) for _ in range(len(recs))]
        recs["lick_tag_bout"] = [np.array([]) for _ in range(len(recs))]
        recs["lick_onset"] = [np.array([]) for _ in range(len(recs))]
        recs["lick_offset"] = [np.array([]) for _ in range(len(recs))]
        recs["lick_tongue_ang_max"] = [np.array([]) for _ in range(len(recs))]

        for this_rec_ind, this_rec in recs.iterrows():
            try:
                tongue = load_tongue_traces(os.path.join(path_ephys, f"data_{this_rec['animal']}"), this_rec["rec"])
                if len(tongue.keys()) == 0:
                    raise FileNotFoundError()
            except FileNotFoundError:
                continue

            recs.at[this_rec_ind, "tongue_time"] = tongue["traces_data"]["time_1K"]
            recs.at[this_rec_ind, "tongue_tip_x"] = tongue["traces_data"]["tip_tongue_x"]
            recs.at[this_rec_ind, "tongue_tip_y"] = tongue["traces_data"]["tip_tongue_y"]
            recs.at[this_rec_ind, "tongue_l_x"] = tongue["traces_data"]["l_tongue_x"]
            recs.at[this_rec_ind, "tongue_l_y"] = tongue["traces_data"]["l_tongue_y"]
            recs.at[this_rec_ind, "tongue_r_x"] = tongue["traces_data"]["r_tongue_x"]
            recs.at[this_rec_ind, "tongue_r_y"] = tongue["traces_data"]["r_tongue_y"]
            recs.at[this_rec_ind, "tongue_mid_x"] = tongue["traces_data"]["mid_tongue_x"]
            recs.at[this_rec_ind, "tongue_mid_y"] = tongue["traces_data"]["mid_tongue_y"]
            recs.at[this_rec_ind, "lick_tag"] = tongue["lick_data"]["tag_lick"]
            recs.at[this_rec_ind, "lick_tag_bout"] = tongue["lick_data"]["tag_bout"]
            recs.at[this_rec_ind, "lick_onset"] = tongue["lick_data"]["time_onset"]
            recs.at[this_rec_ind, "lick_offset"] = tongue["lick_data"]["time_offset"]
            recs.at[this_rec_ind, "lick_tongue_ang_max"] = tongue["lick_data"]["tongue_ang_max"]

    return recs, units
