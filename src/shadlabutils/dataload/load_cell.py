import os
from typing import Dict

import mat73


def load_cell(
    path_data_monkey_sorted: str,
    sess: str,
    cell_id: str,
    flag_eye_traces=False,
    flag_tongue_traces=False,
) -> Dict:
    if len(sess) == 6:
        sess = f"20{sess[:2]}-{sess[2:4]}-{sess[4:6]}"

    sess_name = sess[2:].replace("-", "")

    if len(cell_id) == 34:
        cell_name = cell_id
    elif len(cell_id) == 35:
        cell_name = cell_id
    elif len(cell_id) == 20:
        path_ = os.path.join(
            path_data_monkey_sorted, sess[:7], sess, "units", f"{cell_id}_combine_*.mat"
        )
        cell_name = os.listdir(path_)[0]
    else:
        path_ = os.path.join(
            path_data_monkey_sorted,
            sess[:7],
            sess,
            "units",
            f"{sess_name}_*_{cell_id}_combine_*.mat",
        )
        cell_name = os.listdir(path_)[0]

    path_to_units = os.path.join(path_data_monkey_sorted, sess[:7], sess, "units")
    path_mat = os.path.join(path_to_units, cell_name)

    data = mat73.loadmat(path_mat)
    data["cell_name"] = cell_name[1:-4]

    if flag_eye_traces:
        eye_traces = mat73.loadmat(
            os.path.join(path_to_units, f"{sess_name}_eye_traces.mat")
        )["eye_traces"]
        data["eye_traces"] = {
            k: [v[i] for i, vv in enumerate(data["rec_info"]["rec_flag"]) if vv]
            for k, v in eye_traces.items()
        }

    if flag_tongue_traces:
        tongue_traces = mat73.loadmat(
            os.path.join(path_to_units, f"{sess_name}_tongue_traces.mat")
        )
        data["tongue_traces"] = {
            k: [v[i] for i, vv in enumerate(data["rec_info"]["rec_flag"]) if vv]
            for k, v in tongue_traces.items()
        }

    return data
