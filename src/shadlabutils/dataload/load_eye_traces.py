import os
from typing import Dict

import mat73


def load_eye_traces(path_data_monkey_sorted: str, sess: str) -> Dict:
    if len(sess) == 6:
        sess = f"20{sess[:2]}-{sess[2:4]}-{sess[4:6]}"

    sess_name = sess[2:].replace("-", "")

    path_to_units = os.path.join(path_data_monkey_sorted, sess[:7], sess, "units")
    return mat73.loadmat(os.path.join(path_to_units, f"{sess_name}_eye_traces.mat"))[
        "eye_traces"
    ]
