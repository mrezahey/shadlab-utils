import numpy as np

params = {
    "elec_info": [
        {
            "elec": {
                "x": 16 * np.tile([0, 2, 1, 3], 16),
                "y": 20 * np.repeat(np.arange(1, 33), 2),
                "model": "Cambridge Checkerboard (M1)",
                "length": 0.8e3,  # um
                "level": 40,
            }
        },
        {
            "elec": {
                "x": np.zeros((1, 64)),
                "y": 31 * np.arange(1, 65),
                "model": "Cambridge Linear (M2)",
                "length": 2.1e3,  # um
                "level": 31,
            }
        },
        {
            "elec": {
                "x": 3 * np.array([3**0.5, 3**0.5, (3**0.5) / 2, 3 * (3**0.5) / 2]),
                "y": 6 * np.array([3, 2, 3 / 2, 3 / 2]),
                "model": "tetrode",
            }
        },
        {
            "elec": {
                "x": 3
                * np.array(
                    [
                        0,
                        (3**0.5),
                        (3**0.5) / 2,
                        (3**0.5),
                        3 * (3**0.5) / 2,
                        (3**0.5),
                        2 * (3**0.5),
                    ]
                ),
                "y": 6 * np.array([3, 0, 3 / 2, 2, 3 / 2, 3, 3]),
                "model": "heptode",
            }
        },
        {
            "elec": {
                "x": 16 * np.tile([0, 2, 1, 3], 16 * 6),
                "y": 20 * np.repeat(np.arange(1, 32 * 6 + 1), 2),
                "model": "npx1",
                "length": 4e3,  # um
                "level": 40,
            }
        },
        {
            "elec": {
                "x": 16 * np.tile([0, 2, 1, 3], 16 * 16),
                "y": 20 * np.repeat(np.arange(1, 32 * 16 + 1), 2),
                "model": "npx1 nhp",
                "length": 4e3,  # um
                "level": 40,
            }
        },
    ],
    "sac": {
        "tag_name_list": [
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
        ],
        "align_name_list": [
            "onset",
            "offset",
        ],
    },
}
