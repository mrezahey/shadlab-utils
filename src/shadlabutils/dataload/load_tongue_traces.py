import os
from typing import Dict

import mat73


def load_tongue_traces(path_data_monkey_sorted: str, rec: str) -> Dict:
    """

    :param path_data_monkey_sorted: like "Z:\\Ephys\\data_132F"
    :param rec: like "2023-01-11_15-37-36"
    :return:
    """
    path_to_tongue = os.path.join(
        path_data_monkey_sorted, rec[:7], rec[:10], rec, "analyzed_data", "behavior_data", "tongue"
    )
    analyzed_data_list = [f for f in os.listdir(path_to_tongue) if f.endswith("_ANALYZED.mat")]
    if len(analyzed_data_list) == 0:
        return {}
    elif len(analyzed_data_list) == 1:
        return mat73.loadmat(os.path.join(path_to_tongue, analyzed_data_list[0]))
