# !/usr/bin/env python.
# -*- coding: utf-8 -*-
"""

Name: ELAM-flume-2D
Purpose: simulate the movement of an agent in a confined geometry based
on the response to the incoming flow information


Parameters
----------
Dataframe of the flow field (x, y, uflow, vflow)
Shapefile of the geometry 

Returns
-------
Tracks of the simulated agents
Relation between fatigue and motivation

License
-------
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Author: Abbas El Hachem
Institution: Bundesanstalt für Wasserbau | BAW
Copyright: (c) 2025, Bundesanstalt für Wasserbau | BAW
Email: abbas.el_hachem@baw.de
Version: 0.1
Last Update: 01.05.2025

"""


__author__ = "Abbas El Hachem"
__institution__ = (
    "Bundesanstalt für Wasserbau | BAW" 
    "Hydraulic Engineering in Inland Areas"
)
__copyright__ = "GNU General Public License 3"
__email__ = "abbas.el_hachem@baw.de"
__version__ = 0.1
__last_update__ = "01.05.2025"
# =============================================================================
import os
import tqdm

import pandas as pd
import numpy as np  

from pathlib import Path

# import needed functions from main
from _0_function_ELAM_flume_2D import (read_output_CFD_data_2D, 
                                       update_fish_pos_2D,
                                       create_grid_memory_map,
                                       read_geom_shp,
                                       plot_results_after,
                                       plot_mvt_fish)
# define random seed for reproducibility
np.random.seed(5)


# %%
if __name__ == '__main__':


    # out directory for saving results
    out_save_file = Path(r"ELAM_flume_2D")

    if not os.path.exists(out_save_file):
        os.mkdir(out_save_file)
        
    # define data directory
    data_dir = Path(r"")
    
    path_flow_field = Path(data_dir / 'data/2d_flow_field.csv')

    path_shp = Path(data_dir / r"data/Rhinne_2D.shp")


    # % parameters
    fish_BL = 0.1 # m used in calculation of Fatigue
    
    grid_size_memory_map = 0.2
    
    tend = 3600 #s end time of experiment
    tstep = 0.5 #s fish agent movement step

    nfish_sim = 20 # number of fish agents
    
    # used for output matrix - column names
    cols_fish = ["x", "y", "vel", "ang_h", "u_fish",
                 "v_fish", "f_fish", "m_fish", 'tspot_fish']

    fish_great_flow = 0.25 # percent
    
    add_rdm_agnle = True # to fish swim angle
    min_bound = -15.0
    max_bound = 15.0
    
    delta_xyz_cfd = 0.05  # m

    # parameters for same spot / stuck
    tsepts_for_tspot = 5
    t_stuck = 15 #s
    r_x_same_spot = 1  # m
    r_y_same_spot = 0.5  # m

    # Motivation parameters
    mem_m = 0.9  # 0.92-0.97
    initial_motiv = 0.5  # 0-0.5

    # maximum time needed to reach maximum motivation
    # for fast and slow fish
    k_M_f = 2  # 2-32
    k_M_s = 90  # 90-140
    
    # first are slow then fast
    nbr_slow_fish = int(0.2*nfish_sim)
    nbr_fast_fish = int(nfish_sim - nbr_slow_fish)

    delta_hold = 0.05  # 0.02-0.07

    # fatigue parameters
    initial_fatigue = 0.15
    fatigue_den_kf = 19  # 19-29
    fatigue_dec_coef_mfd = 0.99  # 0.95-1
    fatigue_inc_coef_mfi = 0.3  # 0.24-0.34,

    n_perc_pass = 0.81  # number of fish that pass to stop program
    
    # number of tracks to plot
    nfish_plot = 3
    plot_gif = True
    
    # %% read CFD data and get 2D field
    df_2D_cfd, xy_grid_cfd_tree, xy_points = read_output_CFD_data_2D(
        path_2d_flow_field=path_flow_field,
        xcol_name='xcfd_grid',
        ycol_name='ycfd_grid')
    
    
    # create memory map grid_tree and dataframe
    grid_tree, df_xy_cfd_fish = create_grid_memory_map(
        xmin=min(df_2D_cfd.loc[:, 'xcfd_grid']),
        xmax=max(df_2D_cfd.loc[:, 'xcfd_grid']),
        ymin=min(df_2D_cfd.loc[:, 'ycfd_grid']),
        ymax=max(df_2D_cfd.loc[:, 'ycfd_grid']),
        cell_size=grid_size_memory_map,
        nbr_fish=nfish_sim)

    # read shapefile geometry
    bounds_poly_x, bounds_poly_y, polygon_feature = read_geom_shp(path_shp)

    # call main model function
    for tstep_iter in tqdm.tqdm(np.arange(0, tend + tstep, tstep)):

        if tstep_iter == 0:
            fish_mtx_tm1 = np.zeros(shape=(nfish_sim, len(cols_fish)))
            fish_mtx_tp1 = np.zeros(shape=(nfish_sim, len(cols_fish)))

            # start x, start y
            fish_mtx_tm1[:, :1] = df_2D_cfd.xcfd_grid.max()-0.5
            fish_mtx_tm1[:, 1:2] = np.round(
                np.linspace(df_2D_cfd.ycfd_grid.min(),
                            df_2D_cfd.ycfd_grid.max(), nfish_sim), 1
            ).reshape(fish_mtx_tm1[:, 1:2].shape)

            fish_mtx_tp1 = fish_mtx_tm1

            fish_mtx_tp1[:, 6:7] = initial_fatigue
            fish_mtx_tp1[:, 7:8] = initial_motiv

            mtx_shape = fish_mtx_tp1[:, :1].shape

            df_xynew = pd.DataFrame(
                index=range(len(fish_mtx_tp1[:, :1])))
            df_xynew['x_f'] = fish_mtx_tp1[:, :1]
            df_xynew['y_f'] = fish_mtx_tp1[:, 1:2]

            df_x_results = pd.DataFrame(
                index=range(len(fish_mtx_tp1[:, :1])),
                columns=np.arange(0, tend+tstep, tstep))
            df_y_results = pd.DataFrame(
                index=range(len(fish_mtx_tp1[:, :1])),
                columns=np.arange(0, tend+tstep, tstep))

            df_x_results.iloc[:, 0] = fish_mtx_tp1[:, :1]
            df_y_results.iloc[:, 0] = fish_mtx_tp1[:, 1:2]

            df_u_results = pd.DataFrame(
                index=range(len(fish_mtx_tp1[:, :1])),
                columns=np.arange(0, tend+tstep, tstep))
            df_v_results = pd.DataFrame(
                index=range(len(fish_mtx_tp1[:, :1])),
                columns=np.arange(0, tend+tstep, tstep))

            df_fatigue_results_out = pd.DataFrame(
                index=range(len(fish_mtx_tp1[:, :1])),
                columns=np.arange(0, tend+tstep, tstep))
            df_fatigue_results_out.iloc[:, 0] = fish_mtx_tp1[:, 6:7]

            df_motiv_results_out = pd.DataFrame(
                index=range(len(fish_mtx_tp1[:, :1])),
                columns=np.arange(0, tend+tstep, tstep))
            df_motiv_results_out.iloc[:, 0] = fish_mtx_tp1[:, 7:8]

            df_tspot_results_out = pd.DataFrame(
                index=range(len(fish_mtx_tp1[:, :1])),
                columns=np.arange(0, tend+tstep, tstep))
            df_tspot_results_out.iloc[:, 0] = fish_mtx_tp1[:, 8:9]

            migrating_fish = np.arange(0, nfish_sim, 1, dtype=int)
            drifting_fish = np.array([], dtype=int)
            holding_fish = np.array([], dtype=int)
        else:
            if len(np.where(
                    fish_mtx_tp1[:, :1] < -1.)[0]) < n_perc_pass*nfish_sim:

                (endx_fish,
                    endy_fish,
                    speed_fish,
                    angle_fish_horiz,
                    fatigue_avg,
                    df_u_results,
                    df_v_results,
                    df_xy_cfd_fish,
                    df_x_results,
                    df_y_results,
                    df_fatigue_results_out,
                    df_motiv_results_out,
                    df_tspot_results_out,
                    migrating_fish,
                    holding_fish,
                    drifting_fish) = update_fish_pos_2D(
                    old_fish_pos_mtx=fish_mtx_tp1,
                    xy_grid_tree=xy_grid_cfd_tree,
                    df_x_results=df_x_results,
                    df_y_results=df_y_results,
                    df_u_results=df_u_results,
                    df_v_results=df_v_results,
                    npoint_2d_Umean=df_2D_cfd.loc[:, 'u_f'],
                    npoint_2d_Vmean=df_2D_cfd.loc[:, 'v_f'],
                    flow_field_mag_2d=df_2D_cfd.loc[:, 'U_mag'],
                    fish_BL=fish_BL,
                    migrating_fish=migrating_fish,
                    holding_fish=holding_fish,
                    drifting_fish=drifting_fish,
                    df_fatigue_results=df_fatigue_results_out,
                    df_motiv_results=df_motiv_results_out,
                    df_tspot_results=df_tspot_results_out,
                    polygon_feature=polygon_feature,
                    dt_step=tstep,
                    fish_great_flow=fish_great_flow,
                    add_rdm_agnle=add_rdm_agnle,
                    delta_xyz_cfd=delta_xyz_cfd,
                    xcfd_grid_zlevel=df_2D_cfd.loc[:, 'xcfd_grid'],
                    ycfd_grid_zlevel=df_2D_cfd.loc[:, 'ycfd_grid'],
                    col_val=np.where(
                        df_x_results.columns == tstep_iter)[0][0],
                    xy_points=xy_points,
                    t_stuck=t_stuck,
                    r_x_same_spot=r_x_same_spot,
                    r_y_same_spot=r_y_same_spot,
                    mem_m=mem_m,
                    k_M_f=k_M_f,
                    k_M_s=k_M_s,
                    n_iss=tsepts_for_tspot,
                    nbr_slow_fish=nbr_slow_fish,
                    nbr_fast_fish=nbr_fast_fish,
                    df_xy_cfd_fish=df_xy_cfd_fish,
                    grid_xy=grid_tree,
                    mf_d=fatigue_dec_coef_mfd,
                    mf_i=fatigue_inc_coef_mfi,
                    k_f=fatigue_den_kf,
                    nfish=nfish_sim,
                    min_bound=min_bound,
                    max_bound=max_bound,
                    delta_hold=delta_hold)

                # apped results and send back into function
                fish_mtx_tp1[:, :1] = endx_fish.reshape(mtx_shape)

                fish_mtx_tp1[:, 1:2] = endy_fish.reshape(mtx_shape)

                fish_mtx_tp1[:, 2:3] = speed_fish.reshape(mtx_shape)

                fish_mtx_tp1[:, 3:4] = angle_fish_horiz.reshape(mtx_shape)

                fish_mtx_tp1[:, 6:7] = fatigue_avg.reshape(mtx_shape)

print('simulation ended')
#%% plot results 


plot_results_after(df_cfd_2d=df_2D_cfd, df_x_results=df_x_results,
                   df_y_results=df_y_results,
                   df_fatigue=df_fatigue_results_out,
                   df_motiv=df_motiv_results_out,
                   bounds_poly_x=bounds_poly_x,
                   bounds_poly_y=bounds_poly_y,
                   out_path=out_save_file)

# plot movement vectors
plot_mvt_fish(df_cfd_2d=df_2D_cfd,
              df_x_results=df_x_results,
              df_y_results=df_y_results,
              df_u_fish=df_u_results,
              df_v_fish=df_v_results,
              bounds_poly_x=bounds_poly_x,
              bounds_poly_y=bounds_poly_y,
              ix_plot=15,
              out_path=out_save_file,
              xy_grid_cfd_tree=xy_grid_cfd_tree,
              nfish_plot=nfish_plot,
              dt_step=tstep,
              plot_gif=plot_gif)
