import os

import numpy as np
import pandas as pd

from . import params


def extract_cell_metadata(path_data_monkey_sorted: str, session: str):
    units_info = dict()
    units_info["sess_path"] = os.path.join(
        path_data_monkey_sorted, session[:7], session
    )
    units_info["sess_name"] = session[2:].replace("-", "")

    # read session meta data and all rec meta data
    sess_meta_data = pd.read_excel(
        os.path.join(units_info["sess_path"], f"{units_info['sess_name']}.xls")
    )
    units_info["units_path"] = os.path.join(units_info["sess_path"], "units")
    units_info["rec_list"] = list(
        sess_meta_data.loc[
            (sess_meta_data["ephys"] == 1) & (sess_meta_data["eye"] == 1), "folder_name"
        ]
    )
    units_info["mpm"] = list(
        sess_meta_data.loc[
            (sess_meta_data["ephys"] == 1) & (sess_meta_data["eye"] == 1), "MPM"
        ]
    )
    units_info["num_trial"] = list(
        sess_meta_data.loc[
            (sess_meta_data["ephys"] == 1) & (sess_meta_data["eye"] == 1), "num_trial"
        ]
    )
    units_info["elec"] = params["elec_info"][
        sess_meta_data.loc[
            (sess_meta_data["ephys"] == 1) & (sess_meta_data["eye"] == 1), "elec"
        ].iloc[0]
        - 1
    ]["elec"]

    n_recs = len(units_info["rec_list"])
    units_info["rec_path"] = list()
    recs_bndl = list()
    recs_type = list()
    recs_units = list()
    recs_sac_mod = list()
    recs_lick_mod = list()
    recs_mod = list()
    recs_clique = list()
    rec_meta_data = list()
    # loop over recs and read rec metadata
    for current_rec in units_info["rec_list"]:
        rec_name = current_rec[2:].replace("-", "")
        units_info["rec_path"].append(
            os.path.join(units_info["sess_path"], current_rec)
        )

        # read rec_meta_data
        if not os.path.exists(
            os.path.join(units_info["rec_path"][-1], f"{rec_name}.xls")
        ):
            raise FileNotFoundError("No rec Metadata!")
        rec_meta_data.append(
            pd.read_excel(os.path.join(units_info["rec_path"][-1], f"{rec_name}.xls"))
        )
        recs_bndl.append(rec_meta_data[-1]["cell"].values)
        if np.any(np.isnan(recs_bndl[-1])):
            raise ValueError("Invalid Bundling!")
        recs_units.append(rec_meta_data[-1]["unit_name"].values)
        recs_type.append(rec_meta_data[-1]["type"].values)
        recs_sac_mod.append(
            rec_meta_data[-1]
            .get("sac", pd.Series([False] * recs_type[-1].shape[0]))
            .values
        )
        recs_lick_mod.append(
            rec_meta_data[-1]
            .get("lick", pd.Series([False] * recs_type[-1].shape[0]))
            .values
        )
        recs_mod.append(
            rec_meta_data[-1]
            .get("mod", pd.Series([np.nan] * recs_type[-1].shape[0]))
            .replace(np.nan, "")
            .values
        )
        recs_clique.append(
            rec_meta_data[-1]
            .get("clique", pd.Series([0.0] * recs_type[-1].shape[0]))
            .values
        )

    cell_list_ = os.listdir(os.path.join(units_info["sess_path"], "units"))
    cell_list_ = [i for i in cell_list_ if "_combine_" in i and i.endswith(".mat")]
    units_info["cell_list"] = cell_list_
    units_info["cell_bndl"] = [i[-17:-14] for i in units_info["cell_list"]]

    n_cells = len(units_info["cell_bndl"])

    # find units average positions and types
    units_info["cell_type"] = [None] * n_cells
    units_info["cell_x"] = np.zeros(n_cells)
    units_info["cell_y"] = np.zeros(n_cells)
    units_info["cell_ids"] = [["" for _ in range(n_recs)] for _ in range(n_cells)]
    units_info["cell_sac_mod"] = np.full(n_cells, False)
    units_info["cell_lick_mod"] = np.full(n_cells, False)
    units_info["cell_mod"] = [None] * n_cells
    units_info["cell_clique"] = np.zeros(n_cells)

    for counter_cell, this_cell_bndl in enumerate(units_info["cell_bndl"]):
        inds = [np.where(bn == int(this_cell_bndl))[0] for bn in recs_bndl]
        inds = [i[0] if i.shape[0] > 0 else None for i in inds]
        x_ = list()
        y_ = list()
        for counter_recs, current_unit_ind in enumerate(inds):
            current_mpm = units_info["mpm"][counter_recs]
            if current_unit_ind is None:
                continue
            units_info["cell_type"][counter_cell] = recs_type[counter_recs][
                current_unit_ind
            ]
            units_info["cell_sac_mod"][counter_cell] = recs_sac_mod[counter_recs][
                current_unit_ind
            ]
            units_info["cell_lick_mod"][counter_cell] = recs_lick_mod[counter_recs][
                current_unit_ind
            ]
            units_info["cell_mod"][counter_cell] = recs_mod[counter_recs][
                current_unit_ind
            ]
            units_info["cell_clique"][counter_cell] = recs_clique[counter_recs][
                current_unit_ind
            ]
            current_unit_ch = int(recs_units[counter_recs][current_unit_ind][:-4])
            units_info["cell_ids"][counter_cell][counter_recs] = (
                recs_units[counter_recs][current_unit_ind] + ""
            )
            x_.append(np.squeeze(units_info["elec"]["x"])[current_unit_ch - 1])
            y_.append(current_mpm - np.squeeze(units_info["elec"]["y"])[current_unit_ch - 1])
        units_info["cell_x"][counter_cell] = np.mean(x_)
        units_info["cell_y"][counter_cell] = np.mean(y_)

    return units_info
