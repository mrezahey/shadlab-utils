from typing import List
import os

import mat73
import numpy as np
from scipy.io import loadmat


def compute_calibration_matrix(
    path_data_monkey_sorted: str, session: str, using_tags: List[float] = None
) -> (np.ndarray, np.ndarray):
    """
    Computes the calibration matrix for a given session and path_data_monkey_sorted.
    Note that I used the average raw data of last -200 to -100 indexes to compute
    the raw location of each eye.
    TODO: I used only tag number 4 and did not check the other tags.

    :param path_data_monkey_sorted:
    :param session: like "2024-01-14"
    :param using_tags: list of tags to use for calibration. Default is 4 if None is passed.
    :return: two calibration matrices (left, right). Each matrix has the shape of 3x2
    """
    if using_tags is None:
        using_tags = [4.0]

    all_tgt = list()
    raw_l = list()
    raw_r = list()

    # all recordings of the session
    all_recs = [
        r for r in os.listdir(str(os.path.join(path_data_monkey_sorted, session[:7], session)))
        if r.startswith(session)
    ]

    # loop over all recordings of the session
    for rec in all_recs:
        analyzed_eye_dir = str(
            os.path.join(path_data_monkey_sorted, rec[:7], rec[:10], rec, "analyzed_data", "behavior_data", "eye")
        )
        raw_dir = str(
            os.path.join(path_data_monkey_sorted, rec[:7], rec[:10], rec, "raw_data")
        )

        # load raw data and convert keys to a well-behaved dict
        data = loadmat(
            str(os.path.join(raw_dir, [f for f in os.listdir(raw_dir) if
                                       ("corrective_saccade" in f or "reward_prediction" in f) and ".mat" in f][0],))
        )
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

        # load analyzed data to find the time of correct tags for each trial
        analyzed_data = mat73.loadmat(
            str(os.path.join(analyzed_eye_dir, [f for f in os.listdir(analyzed_eye_dir) if "_ANALYZED.mat" in f][0],))
        )
        tags_ind = np.where(np.isin(analyzed_data["sac_data"]["tag"], using_tags))[0]
        trl_to_time = {
            int(analyzed_data["sac_data"]["trial_num"][tr_num]): analyzed_data["sac_data"]["time_visual"][tr_num]
            for tr_num in tags_ind
        }

        # loop over each trial and extract 1. tgt position
        # 2. left x and y of either left and right eye for the last moments of the trial
        for this_trial_index, trl_start_time in trl_to_time.items():
            if "device_time_data" in data[f"trial_{this_trial_index}"]:
                this_time_ = np.squeeze(data[f"trial_{this_trial_index}"]["device_time_data"])
                candid_l_x = np.squeeze(data[f"trial_{this_trial_index}"]["eye_lx_raw_data"])[this_time_ >= trl_start_time]
                candid_l_y = np.squeeze(data[f"trial_{this_trial_index}"]["eye_ly_raw_data"])[this_time_ >= trl_start_time]
                candid_r_x = np.squeeze(data[f"trial_{this_trial_index}"]["eye_rx_raw_data"])[this_time_ >= trl_start_time]
                candid_r_y = np.squeeze(data[f"trial_{this_trial_index}"]["eye_ry_raw_data"])[this_time_ >= trl_start_time]
            else:
                trl_end_time = np.squeeze(data[f"trial_{this_trial_index}"]["tgt_time_data"])[-1]
                this_time_ = np.squeeze(data["device_time_data"])
                sel_ = (this_time_ >= int(trl_start_time * 1000)) & (this_time_ <= int(trl_end_time * 1000))
                candid_l_x = np.squeeze(data["eye_lx_raw_data"])[sel_]
                candid_l_y = np.squeeze(data["eye_ly_raw_data"])[sel_]
                candid_r_x = np.squeeze(data["eye_rx_raw_data"])[sel_]
                candid_r_y = np.squeeze(data["eye_ry_raw_data"])[sel_]
            this_tgt_x = np.squeeze(data[f"trial_{this_trial_index}"]["tgt_x_data"])[-1]
            this_tgt_y = np.squeeze(data[f"trial_{this_trial_index}"]["tgt_y_data"])[-1]
            all_tgt.append([this_tgt_x, this_tgt_y])
            raw_l.append([np.mean(candid_l_x[-200:-100]), np.mean(candid_l_y[-200:-100])])
            raw_r.append([np.mean(candid_r_x[-200:-100]), np.mean(candid_r_y[-200:-100])])


    all_tgt = np.stack(all_tgt)
    raw_l = np.stack(raw_l)
    raw_r = np.stack(raw_r)

    # get rid of nans
    bad_index_l = np.unique(np.where(np.isnan(raw_l))[0])
    bad_index_r = np.unique(np.where(np.isnan(raw_r))[0])
    raw_l = np.delete(raw_l, bad_index_l, axis=0)
    all_tgt_l = np.delete(all_tgt, bad_index_l, axis=0)
    raw_r = np.delete(raw_r, bad_index_r, axis=0)
    all_tgt_r = np.delete(all_tgt, bad_index_r, axis=0)

    # least square solution
    cal_mat_l, _, _, _ = np.linalg.lstsq(
        np.concatenate([raw_l, np.ones((raw_l.shape[0], 1))], axis=1),
        all_tgt_l,
        rcond=None,
    )
    cal_mat_r, _, _, _ = np.linalg.lstsq(
        np.concatenate([raw_r, np.ones((raw_r.shape[0], 1))], axis=1),
        all_tgt_r,
        rcond=None,
    )

    return cal_mat_l, cal_mat_r
