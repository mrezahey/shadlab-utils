from dataclasses import dataclass
from typing import List

import numpy as np


################################### Electrode info  ##########################################

@dataclass
class ElecInfo:
    x: np.ndarray
    y: np.ndarray
    model: str
    length: float  # um
    level: float


elec_info: List[ElecInfo] = [
    ElecInfo(
        x=16 * np.tile([0, 2, 1, 3], 16),
        y=20 * np.repeat(np.arange(1, 33), 2),
        model="Cambridge Checkerboard (M1)",
        length=0.8e3,
        level=40,
    ),
    ElecInfo(
        x=np.zeros((1, 64)),
        y=31 * np.arange(1, 65),
        model="Cambridge Linear (M2)",
        length=2.1e3,
        level=31,
    ),
    ElecInfo(
        x=3 * np.array([3 ** 0.5, 3 ** 0.5, (3 ** 0.5) / 2, 3 * (3 ** 0.5) / 2]),
        y=6 * np.array([3, 2, 3 / 2, 3 / 2]),
        model="tetrode",
        length=np.nan,
        level=np.nan,
    ),
    ElecInfo(
        x=3 * np.array([0, (3 ** 0.5), (3 ** 0.5) / 2, (3 ** 0.5), 3 * (3 ** 0.5) / 2, (3 ** 0.5), 2 * (3 ** 0.5)]),
        y=6 * np.array([3, 0, 3 / 2, 2, 3 / 2, 3, 3]),
        model="heptode",
        length=np.nan,
        level=np.nan,
    ),
    ElecInfo(
        x=16 * np.tile([0, 2, 1, 3], 16 * 6),
        y=20 * np.repeat(np.arange(1, 32 * 6 + 1), 2),
        model="npx1",
        length=4e3,
        level=40,
    ),
    ElecInfo(
        x=16 * np.tile([0, 2, 1, 3], 16 * 16),
        y=20 * np.repeat(np.arange(1, 32 * 16 + 1), 2),
        model="npx1 nhp",
        length=4e3,
        level=40,
    ),
]

################################### Saccade info  ##########################################
tag_name_list: List[str] = [
    "prim_success",
    "prim_attempt",
    "prim_fail",
    "corr_success",
    "corr_fail",
    "back_center_success",
    "back_center_prim",
    "back_center_irrelev",
    "target_irrelev",
    "other_irrelev",
    "prim_no_corr",
    "db_corr_success",
    "corr_no_db_corr",
    "other_irrelev_visual",
    "back_center_irrelev_visual",
]
