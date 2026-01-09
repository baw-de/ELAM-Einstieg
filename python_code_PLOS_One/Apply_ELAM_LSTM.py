# !/usr/bin/env python.
# -*- coding: utf-8 -*-
"""

Name: 
    Apply_ELAM_LSTM (Fish Behavior Simulation)

Purpose:
    To simulate individual fish movement trajectories within a 2D hydraulic 
    environment. The code integrates Eulerian flow data (CFD) with a Lagrangian 
    agent-based approach, using a pre-trained Long Short-Term Memory (LSTM) 
    neural network to predict fish swimming responses to local velocity 
    and turbulence (k).

Parameters
----------
- Flow Field Data: DDES (Delayed Detached Eddy Simulation) snapshots in .csv 
  or .parquet format containing spatial coordinates, velocity (U, V), and turbulence (k).
- Boundary Geometry: ESRI Shapefiles (.shp) defining physical constraints 
  and the valid simulation zone.
- Initial Positions: A behavior CSV containing starting (x, y) coordinates 
  derived from experimental observations.
- Normalization Constants: Mean and standard deviation values used to scale 
  hydraulic inputs for the LSTM model.
- LSTM Model: A JIT-compiled PyTorch model (.pt) representing the learned 
  behavioral rules.

Returns
-------
- Trajectory Data: A comprehensive dataset containing time-series coordinates 
  and velocity vectors for each simulated agent.
- Visualizations: Static maps of fish tracks, interactive Bokeh HTML plots 
  for individual ID tracking, and animated GIFs showing transient movement.


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
"""
__version__ = 0.1
__last_update__ = '15.12.2025'
# %% define imports
import os
import glob
import time
import timeit
import random
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile
import torch
from scipy.spatial import KDTree, distance as dst
from shapely.geometry import Point, Polygon, LineString

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm


# Settings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
# from torch.cuda.amp import GradScaler, autocast

# Check if a GPU is available, and if so, use it; otherwise, use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_geom_shp(path_geo_shp):
    """ read shapefile of boundary object"""
    shape = shapefile.Reader(path_geo_shp)
    try:
        poly = shape.shape(0).__geo_interface__
    except Exception:
        poly = shape.shape(1).__geo_interface__

    bounds_poly_x = [
        poly["coordinates"][0][i][0]
        for i in range(len(poly["coordinates"][0]))]

    bounds_poly_y = [
        poly["coordinates"][0][i][1]
        for i in range(len(poly["coordinates"][0]))]

    polygon_feature = Polygon(
        [Point(x, y)
         for x, y in zip(bounds_poly_x, bounds_poly_y)])

    return bounds_poly_x, bounds_poly_y, polygon_feature


def read_2d_flow_field(path_flow_field):
    """


    Returns
    -------
    dataframe of 2d flow field.

    """
    in_df_flow_field = pd.read_csv(
        path_flow_field, sep=',', index_col=None, engine='c')
    # create a tree from CFD coordinates
    xy_points = np.c_[in_df_flow_field.loc[:, 'Points:0'],
                      in_df_flow_field.loc[:, 'Points:1']]
    xy_grid_tree = KDTree(xy_points)

    return in_df_flow_field, xy_grid_tree


def find_nearest_cfd_cell(xy_grid_cfd, xy_grid_fish, k=1):
    dist_grid, id_nearest_cfd_grid = xy_grid_cfd.query(
        xy_grid_fish, k=k)
    
    return id_nearest_cfd_grid


def update_pos(xy0, us1, uf1, timestep=1/19):
    """
    update position

    Parameters
    ----------
    xy0 : x or y first position
        in m.
    us1 : uswim or vswim
        in m/s
    uf1 : uflow or vflow
        m/x

    Returns
    -------
    xy1 : new x or y postion
        in m

    """
    xy1 = xy0 + (us1 + uf1)*(timestep)
    return xy1

def check_point_in_poly(xf, yf, poly_shape):
    return poly_shape.contains(Point(xf, yf))


def fish_crosses_poly(xf_old, yf_old, xf_new, yf_new, poly_shape):
    fish_line_str = LineString(
        [Point(xf_old, yf_old), Point(xf_new, yf_new)])
    return poly_shape.crosses(fish_line_str)

#% 
def get_input(xy_fish_start, xy_schlitz,
              in_df_flow_field,
              xy_grid_tree,
              start_u,
              start_v,
              device):
    
    dist_to_Target_all = []
    for _i in range(len(xy_fish_start)):
        xy_ = xy_fish_start[_i]
        dist_to_target = dst.euclidean(xy_, xy_schlitz)
        dist_to_Target_all.append(dist_to_target)
        
    # find nearest CFD cell
    id_nearest_cfd_grid = find_nearest_cfd_cell(
            xy_grid_tree, xy_fish_start)
        
    cfd_flow_vals_start = in_df_flow_field.iloc[
            id_nearest_cfd_grid.ravel(), :]
        
    # get flow values
    u_flow_start = cfd_flow_vals_start.loc[:, 'U:0'].values
    v_flow_start = cfd_flow_vals_start.loc[:, 'U:1'].values
    U_mag_start = np.sqrt(u_flow_start**2 + v_flow_start**2)
    k_start = cfd_flow_vals_start.loc[:, 'k'].values + cfd_flow_vals_start.loc[:, 'resolvedTKE'].values
    
    data_array = np.zeros(shape=(len(xy_fish_start), 6))
    
    data_array[:, 0] = u_flow_start
    data_array[:, 1] = v_flow_start
    data_array[:, 2] = U_mag_start
    data_array[:, 3] = k_start
    # data_array[:, 2] = dist_to_Target_all
    data_array[:, 4] = start_u
    data_array[:, 5] = start_v
    
    model_input = torch.tensor(
        data_array, dtype=torch.float32).to(device)
    
    return model_input


# --- Plotting Functions ---

def plot_static_tracks(df_xtracks, df_ytracks, df_uswim, df_vswim, in_df_flow_field, 
                       bounds_poly_x, bounds_poly_y, out_path, run_name):
    """
    Reproduces the Matplotlib static plot from the original script, 
    including the track extrapolation logic.
    """
    logging.info('----------Plotting Static Tracks------------')
    
    # Uswim = np.sqrt(df_uswim**2 + df_vswim**2)
    plt.ioff()
    fig = plt.figure(figsize=(7, 5), dpi=400)
    ax = fig.add_subplot(111)
    
    # Custom Colormap Logic
    cmap = plt.get_cmap('viridis')
    new_cmap = LinearSegmentedColormap.from_list("custom_viridis", cmap(np.linspace(0.15, 0.99, 256)))
    bounds = [0, 0.2, 0.5, 0.75, 1, 1.25, 1.5, 1.75]
    norm = BoundaryNorm(bounds, new_cmap.N)
    
    # Background Flow Scatter
    im = ax.scatter(
        in_df_flow_field.loc[:, 'Points:0'].astype(float),
        in_df_flow_field.loc[:, 'Points:1'].astype(float),
        c=np.sqrt(in_df_flow_field.loc[:, 'UMean:0'].astype(float)**2 +
                  in_df_flow_field.loc[:, 'UMean:1'].astype(float)**2),
        cmap=new_cmap, marker=',', s=0.6, alpha=0.99, norm=norm
    )
    
    pass_track = 0
    
    # Iterate through tracks
    for ixid, _id in enumerate(df_xtracks.columns):
        if len(df_xtracks.loc[:, _id].dropna(how='all')) > 10:
            xplot = df_xtracks.loc[:, _id].dropna(how='all')
            yplot = df_ytracks.loc[:, _id].dropna(how='all')
            
            # Logic: specific geometry check to extend tracks artificially (from original code)
            if min(xplot[-2:]) < 5.8 and min(yplot[-2:] > 1) and max(yplot[-2:] < 1.2):
                pass_track += 1
                new_x_vals = [_x for _x in xplot]
                new_y_vals = [_y for _y in yplot]
                
                # Extrapolate
                for i in range(2):
                    dx = 5. - float(new_x_vals[-1])
                    dy = 1.15 - float(new_y_vals[-1])
                    distance = np.sqrt(dx**2 + dy**2)
                    step_size = 0.05
                    new_x = new_x_vals[-1] + (dx / distance) * step_size
                    new_y = new_y_vals[-1] + (dy / distance) * step_size
                    new_x_vals.append(new_x)
                    new_y_vals.append(new_y)
                
                xplot = new_x_vals
                yplot = new_y_vals
                ax.plot(xplot, yplot, alpha=0.85, linewidth=0.5, markersize=0.4, marker='.')
            
            # Other conditions for plotting
            elif min(yplot) > 0.5 and max(xplot) > 6 and max(yplot) < 1.5:
                ax.plot(xplot, yplot, alpha=0.85, linewidth=0.5, markersize=0.4, marker='.')
            elif max(yplot) > 1.5:
                # Slight trim
                xplot = df_xtracks.loc[:, _id].dropna(how='all')[:-1]
                yplot = df_ytracks.loc[:, _id].dropna(how='all')[:-1]
                ax.plot(xplot, yplot, alpha=0.85, linewidth=0.5, markersize=0.4, marker='.')

    ax.plot(bounds_poly_x, bounds_poly_y, color='k', alpha=0.7)
    
    cbar = plt.colorbar(im, ticks=bounds, pad=0.02, shrink=0.5, orientation='vertical')
    cbar.solids.set_alpha(1)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(r'$\text{Average flow velocity } \text{ [m s}^{-1}]$', size=8)
    
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_xlim([5.3, 10])
    ax.set_ylim([0, 2.5])
    ax.grid(alpha=0.5)
    ax.set_aspect('equal')
    
    print(f"Pass track count: {pass_track}")
    out_file = out_path / f'tracks_{run_name}_isa.png'
    plt.savefig(out_file, bbox_inches='tight')
    plt.close('all')
    logging.info(f"Saved static plot to {out_file}")



# --- Main Logic ---

def main():
    
    # --- PATH DEFINITIONS ---
    main_proj_path = Path(r"C:\Users\el_hachem\Desktop\ELAM_LSTM")
    out_path = main_proj_path / r"results_test"
    out_path.mkdir(parents=True, exist_ok=True)
    
    path_shp = (main_proj_path / r"data\Geometry\Rhinne_2D_Einstieg.shp")
    path_shp_sim = (main_proj_path / r"data\Geometry\simulation_poly.shp")
    
    path_df_flow_field = (main_proj_path / r"data\2D_cut_0.265_DDES.csv")
    path_ddes = main_proj_path / r"data\DDES_fields"
    path_ddes = r"V:\w1\11_FuE\02_Auffindbarkeit\B3953.01.04.70014_ELAM_Einstieg\02_data\von_Lisa\V03_03_m16_ks0.001_uw0.84_DDES"
    path_norm = main_proj_path / r'data\norm_ELAM_input.csv'
    model_path = main_proj_path / r'results/lstm_seq_model__uvUk_rans.pt'
    
    path_obsv_tracks= main_proj_path / (r"data\Tracks_Versuche_A_mit_CFD.csv")

    logging.info(f"**** Started on {time.asctime()} ****")
    start_timer = timeit.default_timer()

    # Hardware check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # --- Load observed Data to get initial agent positions---

    data_df_obsv = pd.read_csv(path_obsv_tracks, sep=';', index_col=0, engine='c')
    start_xy_obsv = np.array([
        data_df_obsv.iloc[np.where(data_df_obsv.id==_id)[0][0],:].loc[['x', 'y']].values.flatten().astype('float')
        for _id in data_df_obsv.id.unique()
    ])

    # 2. Normalization Data (make sure same as input features)
    norm_df = pd.read_csv(path_norm, sep=';', index_col=0)
    norm_vals = norm_df.loc[['UMean:0', 'UMean:1', 'U_mag', 'kMean', 'u_swim', 'v_swim'], :]

    # 3. Model
    model = torch.jit.load(model_path, map_location=device)
    model.eval()

    # 4. Geometry
    bounds_poly_x, bounds_poly_y, polygon_feature = read_geom_shp(path_shp)
    _, _, polygon_feature_sim = read_geom_shp(path_shp_sim)

    # 5. Initial Flow Field (for Static Plot mostly)
    in_df_flow_field_static, xy_grid_tree_static = read_2d_flow_field(path_flow_field=path_df_flow_field)

    # 6. DDES Files if available read them for example 10 fields
    os.chdir(path_ddes)
    all_ddes_files = np.sort(glob.glob('*.gzip'))
    logging.info(f"Found {len(all_ddes_files)} DDES files.")

    # --- Simulation Setup ---
    # input_size = 6
    # dt = 1/19
    xy_schlitz = np.array((5, 1.2))
    
    num_steps = max(100, len(all_ddes_files)) # generate 100 time steps at least
    
    # Initialization
    num_agents = len(start_xy_obsv)
    
    df_generated_xtracks = pd.DataFrame(index=range(num_steps+1), columns=[f'fish_id_{i}' for i in range(num_agents)], data=np.nan)
    df_generated_ytracks = pd.DataFrame(index=range(num_steps+1), columns=[f'fish_id_{i}' for i in range(num_agents)], data=np.nan)
    df_generated_uflow = pd.DataFrame(index=range(num_steps+1), columns=[f'fish_id_{i}' for i in range(num_agents)], data=np.nan)
    df_generated_vflow = pd.DataFrame(index=range(num_steps+1), columns=[f'fish_id_{i}' for i in range(num_agents)], data=np.nan)
    df_generated_uswim = pd.DataFrame(index=range(num_steps+1), columns=[f'fish_id_{i}' for i in range(num_agents)], data=np.nan)
    df_generated_vswim = pd.DataFrame(index=range(num_steps+1), columns=[f'fish_id_{i}' for i in range(num_agents)], data=np.nan)

    df_generated_xtracks.iloc[0,:] = start_xy_obsv[:, 0]
    df_generated_ytracks.iloc[0,:] = start_xy_obsv[:, 1]

    start_u = [random.uniform(-2., 0.) for _ in range(num_agents)]
    start_v = [random.uniform(0, 2) for _ in range(num_agents)]
    
    ix_cols_consider = df_generated_xtracks.columns
    repeat_flow = np.concatenate([all_ddes_files, all_ddes_files]) 
    
    # Ensure repeat_flow has at least 500 values, concatenating as needed.
    # this is only needed if too few DDES fields are available
    # min_length = 1000
    # if len(all_ddes_files) > 0:
    #     n_repeats = int(np.ceil(min_length / len(all_ddes_files)))
    #     repeat_flow = np.concatenate([all_ddes_files] * n_repeats)
    # else:
    #     repeat_flow = np.array([])
    #     logging.warning("No DDES files found to repeat.")
    

    #%% --- Simulation Loop ---
    
    with torch.no_grad():
        for ix, in_t_flow_field_name in enumerate(repeat_flow):
            if ix > 0 and os.path.exists(in_t_flow_field_name) and len(ix_cols_consider) > 1 and ix < 500: 
                # ix < 500 limit from original
                
                print(f'Step {ix}, Active agents: {len(ix_cols_consider)}')
                
                in_df_flow_field = pd.read_parquet(in_t_flow_field_name)
                
                # Rebuild Tree (As per original logic, though inefficient)
                xy_points = np.c_[in_df_flow_field.loc[:, 'Points:0'], in_df_flow_field.loc[:, 'Points:1']]
                xy_grid_tree = KDTree(xy_points)

                in_df_flow_field.loc[:, 'Umag'] = np.sqrt(
                    in_df_flow_field.loc[:, 'U:0']**2 +
                    in_df_flow_field.loc[:, 'U:1']**2 +
                    in_df_flow_field.loc[:, 'U:2']**2
                )
                # Prepare Input
                current_input = get_input(
                    xy_fish_start=start_xy_obsv,
                    xy_schlitz=xy_schlitz,
                    in_df_flow_field=in_df_flow_field,
                    xy_grid_tree=xy_grid_tree,
                    start_u=start_u,
                    start_v=start_v,
                    device=device
                )

                # Standardize
                current_input = torch.tensor(
                    (current_input.cpu().numpy() - norm_vals.loc[:, 'mean'].values) / 
                    (norm_vals.loc[:, 'std'].values),
                    dtype=torch.float32).to(device)
                
                # Predict 
                model.train() # to include randomness dropout
                output, _ = model(current_input.unsqueeze(1)) 
                output = output.cpu().numpy().squeeze(1)

                # De-standardize output
                output = (output * norm_vals.loc[['u_swim', 'v_swim'], 'std'].values) + norm_vals.loc[['u_swim', 'v_swim'], 'mean'].values
                u_swim, v_swim = output[:, 0], output[:, 1]

                # Previous positions
                x_fish = df_generated_xtracks.loc[ix - 1, ix_cols_consider].values
                y_fish = df_generated_ytracks.loc[ix - 1, ix_cols_consider].values

                # Get Flow for recording
                u_flow_norm = current_input[:, 0].cpu().numpy()
                v_flow_norm = current_input[:, 1].cpu().numpy()
                
                u_flow_orig = (u_flow_norm * norm_vals.loc[['UMean:0'], 'std'].values) + norm_vals.loc[['UMean:0'], 'mean'].values
                v_flow_orig = (v_flow_norm * norm_vals.loc[['UMean:1'], 'std'].values) + norm_vals.loc[['UMean:1'], 'mean'].values

                # Apply Logic Constraints (Hardcoded in original)
                for ii in range(len(u_swim)):
                    if 5.7 <= x_fish[ii] <= 5.9 and y_fish[ii] < 1.0:
                        u_swim[ii] = 0.0

                # Update Position
                new_x_fish = update_pos(xy0=x_fish, us1=u_swim, uf1=u_flow_orig.flatten()) # Ensure shape match
                new_y_fish = update_pos(xy0=y_fish, us1=v_swim, uf1=v_flow_orig.flatten())

                # Store
                df_generated_xtracks.loc[ix, ix_cols_consider] = new_x_fish
                df_generated_ytracks.loc[ix, ix_cols_consider] = new_y_fish
                df_generated_uswim.loc[ix-1, ix_cols_consider] = u_swim
                df_generated_vswim.loc[ix-1, ix_cols_consider] = v_swim
                df_generated_uflow.loc[ix-1, ix_cols_consider] = u_flow_orig.flatten()
                df_generated_vflow.loc[ix-1, ix_cols_consider] = v_flow_orig.flatten()

                # Remove Agents leaving simulation polygon
                idx_remove = []
                agent_labels = list(ix_cols_consider)
                
                for i, (xf, yf) in enumerate(zip(new_x_fish, new_y_fish)):
                    if not polygon_feature_sim.contains(Point([xf, yf])):
                        idx_remove.append(agent_labels[i])

                active_columns = df_generated_xtracks.loc[ix, :].dropna().index
                ix_keep = [_ix for _ix in active_columns if _ix not in idx_remove]

                # Prepare for next step
                new_x_fish = df_generated_xtracks.loc[ix, ix_keep].dropna().values
                new_y_fish = df_generated_ytracks.loc[ix, ix_keep].dropna().values
                u_swim = df_generated_uswim.loc[ix-1, ix_keep].dropna().values
                v_swim = df_generated_vswim.loc[ix-1, ix_keep].dropna().values

                ix_cols_consider = ix_keep
                start_xy_obsv = np.stack([new_x_fish, new_y_fish], axis=1)
                start_u = u_swim
                start_v = v_swim

    #%% --- Post-Processing / Plotting ---
    print('Done simulation plotting and saving results')
    run_name = "uvU_rans_test"

    # 1. Static Tracks Plot
    plot_static_tracks(df_generated_xtracks, df_generated_ytracks, 
                       df_generated_uswim, df_generated_vswim,
                       in_df_flow_field_static, bounds_poly_x, bounds_poly_y,
                       out_path, run_name)

    # 2. CSV Export
    logging.info("Preparing data for export...")
    try:
        def melt_variable(df, var_name):
            return df.reset_index().melt(id_vars='index', var_name='fish_id', value_name=var_name)
        
        x_long = melt_variable(df_generated_xtracks.dropna(how='all', axis=0), 'x')
        y_long = melt_variable(df_generated_ytracks.dropna(how='all', axis=0), 'y')
        uswim_long = melt_variable(df_generated_uswim.dropna(how='all', axis=0), 'u_swim')
        vswim_long = melt_variable(df_generated_vswim.dropna(how='all', axis=0), 'v_swim')
        uflow_long = melt_variable(df_generated_uflow.dropna(how='all', axis=0), 'u_flow')
        vflow_long = melt_variable(df_generated_vflow.dropna(how='all', axis=0), 'v_flow')
        
        track_data = (
            x_long
            .merge(y_long, on=['index', 'fish_id'])
            .merge(uswim_long, on=['index', 'fish_id'])
            .merge(vswim_long, on=['index', 'fish_id'])
            .merge(uflow_long, on=['index', 'fish_id'])
            .merge(vflow_long, on=['index', 'fish_id'])
            .rename(columns={'index': 'time'})
        )
        
        csv_out = out_path / f'sim_tracks_{run_name}.csv'
        track_data.to_csv(csv_out, sep=';')
        logging.info(f"Saved tracks CSV to {csv_out}")
        
    except Exception as e:
        logging.error(f"Failed during CSV export: {e}")

    logging.info(f"Finished. Runtime: {timeit.default_timer() - start_timer:.2f}s")

if __name__ == '__main__':
    main()
# =============================================================================
