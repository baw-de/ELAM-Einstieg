# !/usr/bin/env python.
# -*- coding: utf-8 -*-
"""

Name: functions used in the ELAM-flume-2D model
Purpose: model the movement of the agents and plot the results


Parameters
----------
see main file _1_run_ELAM_flume_2D


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
Copyright: (c) 2025, Hydraulic Engineering in Inland Areas
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
import math
import random
import tqdm

import pandas as pd
import numpy as np  
import shapefile

import imageio
import matplotlib.pyplot as plt
from numba import vectorize, float64 

from scipy.spatial import KDTree

from shapely.geometry import Point, Polygon, LineString
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

np.random.seed(5)

# =============================================================================
#
# =============================================================================


def read_geom_shp(path_geo_shp):
    """ read path to boundary object 
        as shapefile (.shp)        
    """
    
    shape = shapefile.Reader(path_geo_shp)
    poly = shape.shape(0).__geo_interface__

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


@vectorize([float64(float64, float64)])
def calc_angle_horiz(u_m, v_m):
    """ calculate angle in degree betweem 2 vectors"""
    angle_2d = math.degrees(math.atan2(v_m, u_m))
    return angle_2d


def read_output_CFD_data_2D(path_2d_flow_field,
                            xcol_name='xcfd_grid',
                            ycol_name='ycfd_grid'
                            ):
    """
    read 2D flow field and return coordinates as tree
    columns format:
        'xcfd_grid', 'ycfd_grid', 'u_f', 'v_f', 'U_mag', 'flow_angle'
    ------

    """
    print('Reading 2D flow field:\n %s'
          % path_2d_flow_field)
    in_df_flow_field = pd.read_csv(path_2d_flow_field,
                                   index_col=0,
                                   sep=";",
                                   engine="c")
    
    # create a tree from CFD coordinates
    xy_points = np.c_[in_df_flow_field.loc[:, xcol_name],
                      in_df_flow_field.loc[:, ycol_name]]
    xy_grid_tree = KDTree(xy_points)

    return in_df_flow_field, xy_grid_tree, xy_points


def create_grid_memory_map(xmin, xmax,
                           ymin, ymax,
                           cell_size, nbr_fish):
    """ create a grid for saving fish positions
        sort of a memory map for every fish
    """
    gridx_for_pos = np.arange(xmin, xmax, cell_size)
    gridy_for_pos = np.arange(ymin, ymax, cell_size)

    grid_x, grid_y = np.meshgrid(gridx_for_pos, gridy_for_pos)
    grid_xy_points = np.c_[grid_x.ravel(), grid_y.ravel()]
    grid_tree = KDTree(grid_xy_points)

    df_xy_cfd_fish = pd.DataFrame(
        index=range(len(grid_xy_points)),
        columns=['fish_id_%d' % nf for nf in range(nbr_fish)],
        data=0)

    return grid_tree, df_xy_cfd_fish


@vectorize([float64(float64, float64)])
def fish_move_mag(flow_mag_at_fish, perc_incr=0.2):
    """
    Adjusts the fish's movement magnitude based on the flow magnitude 
    with a random increment or decrement.

    Parameters:
    flow_mag_at_fish (float): The magnitude of the flow at the fish's location.
    perc_incr (float, optional): The percentage increment or decrement 
    (default is 0.2).

    Returns:
    float: The adjusted movement magnitude for the fish.

    Example:
    >>> fish_move_mag(10)
    12.0
    """
    
    rdm_nbr = np.random.uniform(0, 1, 1)
    if rdm_nbr > 0.5:
        return flow_mag_at_fish + perc_incr * flow_mag_at_fish
    else:
        return flow_mag_at_fish - perc_incr * flow_mag_at_fish

@vectorize([float64(float64)])
def move_against_flow(flow_angle_at_fish):
    """
    Computes the angle for the fish to swim against the flow, 
    given the flow's angle.

    Parameters:
    flow_angle_at_fish (float): The angle of the flow relative 
    to the fish's position in degrees.

    Returns:
    float: The angle the fish should swim to move against the flow.

    Example:
    >>> move_against_flow(45)
    -135.0
    """
    
    if flow_angle_at_fish <= 0:
        opp_angle = 180 - abs(flow_angle_at_fish)
    elif flow_angle_at_fish > 0:
        opp_angle = abs(flow_angle_at_fish) - 180

    return opp_angle


@vectorize([float64(float64, float64, float64)])
def add_random_angle(fish_mov_ang, min_bound, max_bound):
    """
    Adds a random angle to the fish's movement angle, 
    and normalizes the result 
    to the range of -180° to 180°.

    Parameters:
    fish_mov_ang (float): The current movement
    angle of the fish in degrees.
    min_bound (float): The minimum bound of the random angle to add.
    max_bound (float): The maximum bound of the random angle to add.

    Returns:
    float: The new movement angle, normalized to the range of -180° to 180°.

    Example:
    >>> add_random_angle(45, -10, 10)
    50.0
    """
    
    new_rdm_agnle = fish_mov_ang + random.randrange(
        int(min_bound), int(max_bound), 1)

    if new_rdm_agnle <= -180:
        corrct_angle = 360 - abs(new_rdm_agnle)
    elif new_rdm_agnle >= 180:
        corrct_angle = abs(new_rdm_agnle) - 360
    elif -180 < new_rdm_agnle < 180:
        corrct_angle = new_rdm_agnle

    return corrct_angle



@vectorize([float64(float64)])
def make_angle_bet_p_m_pi(ang):
    """
    Normalizes an angle to be between -180° and 180°.

    Parameters:
    ang (float): Angle in degrees.

    Returns:
    float: Normalized angle in the range of -180° to 180°.
    
    Description:
    This function takes an angle and normalizes it so that it lies within 
    the range of -180° to 180°. Angles greater than 180° are mapped to 
    negative angles, and angles less than -180° are mapped to positive angles.

    Example:
    >>> make_angle_bet_p_m_pi(200)
    -160.0
    """
    
    if ang <= -180:
        corrct_angle = 360 - abs(ang)
    elif ang >= 180:
        corrct_angle = abs(ang) - 360
    elif -180 < ang < 180:
        corrct_angle = ang
    return corrct_angle

def calc_fish_swim_vec(angle_dir_fish, fish_speed):
    """
    Calculates the x and y components of a fish's swimming velocity vector 
    based on its direction and speed.

    Parameters:
    angle_dir_fish (float): Direction of the fish's swimming in degrees. 
                             Valid range: -180° to 180°.
    fish_speed (float): Speed of the fish (distance per time).

    Returns:
    tuple: (x_fish, y_fish) components of the fish's velocity.

    Description:
    The function computes the velocity vector components (x and y) using basic
    trigonometric relations based on the fish's direction and speed. It handles 
    all four quadrants of the angle system, adjusting the velocity components 
    accordingly.

    Example:
    >>> calc_fish_swim_vec(45, 2)
    (1.4142135623730951, 1.4142135623730951)
    """
    
    if 0.0 <= angle_dir_fish <= 90.0:
        angle_rad = angle_dir_fish * (math.pi / 180.0)
        x_fish = fish_speed * math.cos(angle_rad)
        y_fish = fish_speed * math.sin(angle_rad)
    elif 90.0 <= angle_dir_fish <= 180.0:
        angle_rad = (180.0 - angle_dir_fish) * (math.pi / 180.0)
        x_fish = -fish_speed * math.cos(angle_rad)
        y_fish = fish_speed * math.sin(angle_rad)
    elif -90.0 <= angle_dir_fish <= 0.0:
        angle_rad = -angle_dir_fish * (math.pi / 180.0)
        x_fish = fish_speed * math.cos(angle_rad)
        y_fish = -fish_speed * math.sin(angle_rad)
    elif -180.0 <= angle_dir_fish <= -90.0:
        angle_rad = (180.0 + angle_dir_fish) * (math.pi / 180.0)
        x_fish = -fish_speed * math.cos(angle_rad)
        y_fish = -fish_speed * math.sin(angle_rad)

    return x_fish, y_fish

def get_df_HN_at_fish_loc(nfish, ngbrs, hn_vals):
    """
    Creates a DataFrame representing the HN values at the fish locations.
    """
    df_hn_f = pd.DataFrame(
        index=range(nfish), columns=range(ngbrs), data=hn_vals)
    return df_hn_f


def meaus_dist_bound(xf, yf, poly_shape):
    """
    Calculates the distance from a point to the boundary of a polygon.

    Parameters:
    xf (float): The x-coordinate of the point.
    yf (float): The y-coordinate of the point.
    poly_shape (shapely.geometry.Polygon): 
        The polygon to calculate the distance to.

    Returns:
    float: The distance from the point to the boundary of the polygon.
    """
    
    return poly_shape.boundary.distance(Point(xf, yf))


def check_point_in_poly(xf, yf, poly_shape):
    """
    Checks if a point is inside a polygon.

    Parameters:
    xf (float): The x-coordinate of the point.
    yf (float): The y-coordinate of the point.
    poly_shape (shapely.geometry.Polygon): 
        The polygonal shape to check against.

    Returns:
    bool: True if the point is inside the polygon, otherwise False.
    """
    
    return poly_shape.contains(Point(xf, yf))


def fish_crosses_poly(xf_old, yf_old, xf_new, yf_new, poly_shape):
    """
    Checks if the fish's movement crosses a polygonal shape.

    Parameters:
    xf_old (float): The x-coordinate of the fish's old position.
    yf_old (float): The y-coordinate of the fish's old position.
    xf_new (float): The x-coordinate of the fish's new position.
    yf_new (float): The y-coordinate of the fish's new position.
    poly_shape (shapely.geometry.Polygon): 
        The polygonal shape to check against.

    Returns:
    bool: True if the fish's movement crosses the polygonal shape,
    otherwise False.
    """
    
    fish_line_str = LineString([Point(xf_old, yf_old), Point(xf_new, yf_new)])
    return poly_shape.crosses(fish_line_str)


@vectorize([float64(float64, float64, float64, float64, float64, float64)])
def calculate_fatigue(us_prev,
                      fn_prev_avg,
                      mf_d,
                      mf_i,
                      fish_BL,
                      k_f):
    """
    Calculate fatigue level based on fish swim speed relative to flow.

    Args:
    - us_prev (float): Swim speed from previous time step (BL/s).
    - fn_prev_avg (float): Previous average fatigue.
    - k_f (float): Fatigue denominator (BL/s) [19-29]
    - mf_d (float): Fatigue memory coefficient when Fn <= Fn-1avg.
    - mf_i (float): Fatigue memory coefficient when Fn > Fn-1avg.

    Returns:
    - fatigue_level (float): Fatigue level in the range [0, 1].
    - 0 means: no fatigue
    - 1 means: exhausted
    """
    # us_prev = 1
    us_prev_bl = us_prev / fish_BL
    # Calculate Fn based on previous time step
    fn = (1.0/k_f) * us_prev_bl

    if fn > 1:
        fn = 1
    # Calculate fatigue memory coefficient based on Fn and Fn-1avg
    if fn <= fn_prev_avg:
        mf = mf_d
    else:
        mf = mf_i

    # Calculate average fatigue based on current and previous fatigue levels
    fn_avg = (1.0 - mf) * fn + mf * fn_prev_avg

    # Ensure fatigue level is within [0, 1] range
    fatigue_level = fn_avg
    if fatigue_level > 1:
        fatigue_level = 1

    assert 0 <= fatigue_level <= 1, 'outside bounds'
    return fatigue_level


@vectorize([float64(float64, float64, float64, float64)])
def calc_eucl_dist(x_new, y_new, x_old, y_old):
    """
    Calculates the Euclidean distance between two points.

    Parameters:
    x_new (float): The x-coordinate of the new point.
    y_new (float): The y-coordinate of the new point.
    x_old (float): The x-coordinate of the old point.
    y_old (float): The y-coordinate of the old point.

    Returns:
    float: The Euclidean distance between the new and old points.
    """
    
    return np.sqrt((x_new - x_old)**2 + (y_new - y_old)**2)

# vectorize the function to receive a vector instead of a single value
vc = np.vectorize(calc_fish_swim_vec)


def update_fish_pos_2D(old_fish_pos_mtx, # matrix with positions t-1
                       xy_grid_tree, # cfd coordinates as tree           
                       df_x_results, # fish x coordinates          
                       df_y_results, # fish y coordinates 
                       df_u_results, # fish uswim velocity             
                       df_v_results, # fish vswim velocity           
                       npoint_2d_Umean, # uflow as pandas series
                       npoint_2d_Vmean, # vflow as pandas series
                       flow_field_mag_2d, # Umag as pandas series
                       fish_BL, # fish body length
                       migrating_fish, # index of migrating fish
                       holding_fish, # index of holding fish
                       drifting_fish, # index of drifting fish
                       df_fatigue_results, # save fatigue values
                       df_motiv_results, # save motivation values
                       df_tspot_results, # save time at a spot 
                       polygon_feature, # polygon geometry 
                       dt_step, # time step for movement
                       fish_great_flow, # percentage for swim speed
                       add_rdm_agnle, # bool for swim vector
                       delta_xyz_cfd, # size of cfd grid
                       xcfd_grid_zlevel, # cfd x coordinates
                       ycfd_grid_zlevel, # cfd y coordinates
                       col_val, # current step iteration
                       xy_points,
                       # t_consider,
                       t_stuck, # time for stuck as a postion
                       r_x_same_spot, # radius in x direction for stuck
                       r_y_same_spot, # radius in y direction for stucl
                       mem_m, # motivation parameter
                       k_M_f, # motivation for strong fish
                       k_M_s, # motivation for slow fish
                       n_iss, # number of steps to consider for tstuck
                       nbr_slow_fish, # number of slow fish
                       nbr_fast_fish, # number of fast fish
                       grid_xy, # 2d grid for map coordinates
                       df_xy_cfd_fish, # dataframe with +1 for fish postion
                       mf_d, # parameter for fatigue dynamics
                       mf_i, #  initial state of fatigue
                       k_f, # Fatigue scaling factor
                       nfish, # number of fish in simulation
                       min_bound, # minimum bound for generating random angle
                       max_bound, # maximum bound for generating random angle
                       delta_hold, # delta time for holding at one position
                       ):
    """
    Updates the position and movement of fish in a 2D grid 
    based on flow field data, fish behavior models, and external conditions.
    
    This function simulates the movement of fish 
    in response to the surrounding flow field, 
    incorporating behavioral factors
    like migrating, holding, and drifting behaviors. 
    The function also applies movement corrections, handles fish interactions,
    and updates their locations within the defined polygon area.

    """
    
    df_xynew = pd.DataFrame(
        index=range(len(old_fish_pos_mtx[:, :1])))
    df_xynew['x_f'] = old_fish_pos_mtx[:, :1]
    df_xynew['y_f'] = old_fish_pos_mtx[:, 1:2]

    xy_coords = df_xynew.dropna(how='any', axis=0)

    dist_grid, id_nearest_cfd_grid = xy_grid_tree.query(
        xy_coords.values, k=1)

    # =========================================================================
    # flow ang and mag at fish loc
    fish_Umean = npoint_2d_Umean.values[
        id_nearest_cfd_grid].ravel().astype("float32")
    fish_Vmean = npoint_2d_Vmean.values[
        id_nearest_cfd_grid].ravel().astype("float32")

    agl_at_fish_loc = calc_angle_horiz(fish_Umean, fish_Vmean)

    mag_at_fish_loc = np.sqrt(fish_Umean**2 + fish_Vmean**2)

    # =========================================================================
    #     # behavior Migrating
    # =========================================================================

    fish_angle_mig = move_against_flow(agl_at_fish_loc[migrating_fish])

    fish_mov_mag_mig = fish_move_mag(
        mag_at_fish_loc[migrating_fish], fish_great_flow)

    # =========================================================================
    #     # behavior Holding
    # =========================================================================
    # usim = uflow
    # swim angle = - flow angle

    if len(holding_fish) > 0:
        fish_angle_holding = move_against_flow(agl_at_fish_loc[holding_fish])
        # 0.01 is arbitrary
        fish_mov_mag_holding = 0.01*mag_at_fish_loc[holding_fish]

    else:
        # migrating behavior
        fish_angle_holding = move_against_flow(
            agl_at_fish_loc[holding_fish])

        fish_mov_mag_holding = fish_move_mag(
            mag_at_fish_loc[holding_fish], fish_great_flow)

    # =========================================================================
    #     # behavior drifting
    # =========================================================================
    # uswim = uflow*[0,1]
    # swim angle = angle flow +- rdm
    # prevent fish from exiting at start area
    if len(drifting_fish) > 0 and min(
            df_x_results.iloc[drifting_fish, col_val-1]) > max(
                xcfd_grid_zlevel) - 0.5:
        # old_fish_pos_mtx[:, : 1].ravel()[drifting_fish]) < 0.2:
        # after 5 time steps they can go downstream
        fish_angle_drifting = agl_at_fish_loc[drifting_fish]
        fish_mov_mag_drifting = mag_at_fish_loc[
            drifting_fish]*np.random.uniform(0.95)
    else:
        # migrating behavior
        fish_angle_drifting = move_against_flow(
            agl_at_fish_loc[drifting_fish])

        fish_mov_mag_drifting = fish_move_mag(
            mag_at_fish_loc[drifting_fish], fish_great_flow)

    # =========================================================================
    #   # append results to one array
    # =========================================================================
    fish_swim_angle = np.zeros(shape=nfish)
    fish_swim_angle[migrating_fish] = fish_angle_mig
    fish_swim_angle[holding_fish] = fish_angle_holding
    fish_swim_angle[drifting_fish] = fish_angle_drifting

    fish_swim_mag = np.zeros(shape=nfish)
    fish_swim_mag[migrating_fish] = fish_mov_mag_mig
    fish_swim_mag[holding_fish] = fish_mov_mag_holding
    fish_swim_mag[drifting_fish] = fish_mov_mag_drifting

    # add random angle to fish angle
    if add_rdm_agnle:
        fish_swim_angle = add_random_angle(
            fish_swim_angle, min_bound, max_bound)

    unew, vnew = vc(angle_dir_fish=fish_swim_angle,
                    fish_speed=fish_swim_mag)

    endx_fish = old_fish_pos_mtx[
        :, :1].ravel() + unew * dt_step
    endy_fish = old_fish_pos_mtx[
        :, 1:2].ravel() + vnew * dt_step

    # anti stuck behavior
    min_x = df_x_results.iloc[:, col_val-n_iss:col_val].min(axis=1)
    max_x = df_x_results.iloc[:, col_val-n_iss:col_val].max(axis=1)
    min_y = df_y_results.iloc[:, col_val-n_iss:col_val].min(axis=1)
    max_y = df_y_results.iloc[:, col_val-n_iss:col_val].max(axis=1)

    # fish at same spot
    idx1 = ((abs(max_x - min_x) < r_x_same_spot) &
            (abs(max_y - min_y) < r_y_same_spot))

    # tspot updated
    t_spot = df_tspot_results.iloc[:, col_val-1].copy(deep=True)
    t_spot[idx1] = t_spot[idx1] + dt_step
    t_spot[~idx1] = 0.

    df_tspot_results.iloc[:, col_val] = t_spot

    # anti stuck move fish add random x,y values
    ix_tstuck = np.where(t_spot >= t_stuck)[0]

    endx_fish[ix_tstuck] = (endx_fish[ix_tstuck] +
                            np.random.uniform(
        -0.35, 0.35, len(endx_fish[ix_tstuck])))

    endy_fish[ix_tstuck] = (endy_fish[ix_tstuck] +
                            np.random.uniform(
        -0.25, 0.25, len(endx_fish[ix_tstuck])))

    # keep fish in polygon
    df_xynew = pd.DataFrame(index=range(len(endx_fish)))
    df_xynew['x_f_n_1'] = old_fish_pos_mtx[:, :1].ravel()
    df_xynew['y_f_n_1'] = old_fish_pos_mtx[:, 1:2].ravel()
    df_xynew['x_f_n_2'] = endx_fish
    df_xynew['y_f_n_2'] = endy_fish

    df_xynew['in_poly'] = df_xynew.apply(lambda x: check_point_in_poly(
        x.x_f_n_2, x.y_f_n_2, polygon_feature), axis=1).values

    ix_not_in_poly = np.where(df_xynew['in_poly'] == False)[0]

    df_near_slot = df_xynew[df_xynew['x_f_n_2'] < 1]
    if len(df_near_slot.index) > 0:

        vals_crossing = df_near_slot.apply(
            lambda x: fish_crosses_poly(x.x_f_n_1, x.y_f_n_1,
                                        x.x_f_n_2, x.y_f_n_2,
                                        polygon_feature), axis=1)
        ix_crosses_poly = vals_crossing.index[
            np.where((vals_crossing == True))[0]]

    else:
        ix_crosses_poly = np.array([], dtype=int)
    while len(ix_not_in_poly) > 0 or len(ix_crosses_poly) > 0:
        # switch direction
        ix_not_in_poly_pro = np.sort(
            np.concatenate([ix_not_in_poly, ix_crosses_poly]))
        fish_swim_angle[ix_not_in_poly_pro] = (
            fish_swim_angle[ix_not_in_poly_pro] +
            int(np.random.uniform(low=min_bound*2, high=max_bound*2)))

        fish_swim_angle[ix_not_in_poly_pro] = make_angle_bet_p_m_pi(
            fish_swim_angle[ix_not_in_poly_pro])

        unew, vnew = vc(
            angle_dir_fish=fish_swim_angle[ix_not_in_poly_pro],
            fish_speed=fish_swim_mag[ix_not_in_poly_pro]*1.25)

        endx_fish[ix_not_in_poly_pro] = (
            old_fish_pos_mtx[:, :1].ravel()[ix_not_in_poly_pro] +
            unew * dt_step)
        endy_fish[ix_not_in_poly_pro] = old_fish_pos_mtx[:, 1:2].ravel()[
            ix_not_in_poly_pro] + vnew * dt_step

        df_xynew = pd.DataFrame(index=range(len(endx_fish)))
        df_xynew['x_f_n_1'] = old_fish_pos_mtx[:, :1].ravel()
        df_xynew['y_f_n_1'] = old_fish_pos_mtx[:, 1:2].ravel()
        df_xynew['x_f_n_2'] = endx_fish
        df_xynew['y_f_n_2'] = endy_fish

        df_xynew['in_poly'] = df_xynew.apply(lambda x: check_point_in_poly(
            x.x_f_n_2, x.y_f_n_2, polygon_feature), axis=1).values

        ix_not_in_poly = np.where(df_xynew['in_poly'] == False)[0]

        if len(df_near_slot.index) > 0:
            df_near_slot = df_xynew[df_xynew['x_f_n_2'] < 1]
            vals_crossing = df_near_slot.apply(
                lambda x: fish_crosses_poly(
                    x.x_f_n_1, x.y_f_n_1, x.x_f_n_2,
                    x.y_f_n_2, polygon_feature), axis=1)

            ix_crosses_poly = vals_crossing.index[
                np.where((vals_crossing == True))[0]]
        else:
            ix_crosses_poly = np.array([], dtype=int)

    # append locations to final dataframe
    df_x_results.iloc[:, col_val] = endx_fish
    df_y_results.iloc[:, col_val] = endy_fish

    # fish vector component in x,y directions m/s

    df_u_results.iloc[:, col_val] = (
        endx_fish-old_fish_pos_mtx[:, :1].ravel() - fish_Umean*dt_step)
    df_v_results.iloc[:, col_val] = (
        endy_fish - old_fish_pos_mtx[:, 1:2].ravel() - fish_Vmean*dt_step)
    
    # calculate map of locations
    endxy_fish = np.array([(x, y) for x, y in zip(endx_fish, endy_fish)])

    dist_grid_n, id_nearest_cfd_grid_n = grid_xy.query(
        endxy_fish, k=1)

    df_xy_cfd_fish.iloc[id_nearest_cfd_grid_n, :] += 1

    # Motivation

    # slow fish
    motivation_n_slow = t_spot[:nbr_slow_fish] / k_M_s
    m_avg_n_slow = (1-mem_m)*motivation_n_slow + (
        mem_m*df_motiv_results.iloc[:nbr_slow_fish, col_val-1])

    # fast fish
    motivation_n_fast = t_spot[nbr_slow_fish:] / k_M_f
    m_avg_n_fast = (1-mem_m)*motivation_n_fast + (
        mem_m*df_motiv_results.iloc[nbr_slow_fish:, col_val-1])

    m_avg_n_slow[m_avg_n_slow > 1] = 1
    m_avg_n_fast[m_avg_n_fast > 1] = 1

    df_motiv_results.iloc[:nbr_slow_fish, col_val] = m_avg_n_slow
    df_motiv_results.iloc[nbr_slow_fish:, col_val] = m_avg_n_fast

    # calculate Fatigue
    avg_fatigue_vals = np.round(calculate_fatigue(
        old_fish_pos_mtx[:, 2:3], old_fish_pos_mtx[:, 6:7],
        mf_d, mf_i, fish_BL, k_f), 3)

    df_fatigue_results.iloc[:, col_val] = avg_fatigue_vals

    # select behavior

    migrating_fish = np.where(df_motiv_results.iloc[:, col_val] >
                              df_fatigue_results.iloc[:, col_val] + delta_hold
                              )[0]

    holding_fish = np.where((df_motiv_results.iloc[:, col_val] >=
                             df_fatigue_results.iloc[:, col_val] - delta_hold)
                            &
                            (df_motiv_results.iloc[:, col_val] <=
                             df_fatigue_results.iloc[:, col_val] + delta_hold)
                            )[0]

    drifting_fish = np.where(df_motiv_results.iloc[:, col_val] <
                             df_fatigue_results.iloc[:, col_val] - delta_hold
                             )[0]

    print('\nmigrating_fish ', len(migrating_fish),
          '\nholding_fish', len(holding_fish),
          '\ndrifting_fish ', len(drifting_fish))

    return [endx_fish, endy_fish,
            fish_swim_mag,
            fish_swim_angle,
            avg_fatigue_vals,
            df_u_results,
            df_v_results,
            df_xy_cfd_fish,
            df_x_results,
            df_y_results,
            df_fatigue_results,
            df_motiv_results,
            df_tspot_results,
            migrating_fish,
            holding_fish,
            drifting_fish]



def plot_results_after(df_cfd_2d, df_x_results, df_y_results,
                       df_fatigue, df_motiv,
                       bounds_poly_x, bounds_poly_y,
                       out_path):
    
    """
    Plots the results of fish movement and related metrics
    such as flow magnitude, fatigue, and motivation over time.
    
    The function generates two plots:
    1. A 2D scatter plot showing the flow field and fish trajectories.
    2. A plot of fatigue and motivation for all fish over time.


    """
    
    print('Plotting results')
    min_mag = df_cfd_2d.loc[:, 'U_mag'].min()
    max_mag = df_cfd_2d.loc[:, 'U_mag'].max()

    bound_count = np.linspace(min_mag, max_mag, 4)
    norm_ppt = plt.Normalize(min(bound_count), max(bound_count))
    cmap_ppt = plt.get_cmap("plasma_r")
    plt.ioff()
    fig = plt.figure(figsize=(12, 4), dpi=300)
    ax = fig.add_subplot(111)

    # Plot cfd 
    im1 = ax.scatter(
        x=df_cfd_2d.loc[:, 'xcfd_grid'],
        y=df_cfd_2d.loc[:, 'ycfd_grid'],
        c=df_cfd_2d.loc[:, 'U_mag'],
        marker="x",
        cmap=cmap_ppt,
        norm=norm_ppt,
        alpha=0.5,
        s=3)

    ax.plot(bounds_poly_x, bounds_poly_y, c="k")

    for nf in df_x_results.index:
        # nf=4
        ax.plot(df_x_results.loc[nf, :],
                df_y_results.loc[nf, :],
                marker='.', alpha=0.35, linewidth=0.71,
                c='k', markersize=2)

    cb = fig.colorbar(
        im1,
        ax=ax,
        label="U mag. (m/s)",
        ticks=np.round(bound_count, 3),
        shrink=0.75,
        extend=None,
        orientation="vertical",
        pad=0.1)
    cb.solids.set_alpha(1)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    # ax.axis('equal')
    ax.set_xlim([min(bounds_poly_x)-0.1, max(bounds_poly_x)+0.1])
    ax.set_ylim([min(bounds_poly_y)-0.1, max(bounds_poly_y)+0.1])
    fig.draw_without_rendering()
    ax.grid(alpha=0.5)
    # ax.set_zlabel('Z')
    plt.savefig(
        out_path /
        r"sim_tracks_Umag.png",
        bbox_inches="tight",
    )
    plt.close("all")

    # plot fatigue motivation
    plt.ioff()
    fig = plt.figure(figsize=(6, 4), dpi=300)
    ax = fig.add_subplot(111)

    df_fatigue.columns = range(1, len(df_fatigue.columns)+1)
    df_motiv.columns = range(1, len(df_motiv.columns)+1)
    for f_id in df_fatigue.index:
        f_ser = df_fatigue.iloc[f_id, :]
        m_set = df_motiv.iloc[f_id, :]
        ax.plot(f_ser.index, f_ser.values, alpha=0.5, linestyle='-.', c='r')
        ax.plot(m_set.index, m_set.values, alpha=0.5, c='k')

    ax.plot(f_ser.index, f_ser.values, alpha=0.5,
            label='Fatigue', linestyle='-.', c='r', marker='o')
    ax.plot(m_set.index, m_set.values, alpha=0.5, label='Motivation', c='k',
            marker='x')
    ax.set_xlabel("Model Timestep")
    ax.set_ylabel("Fatigue/Motivation")
    ax.legend(loc=0)
    ax.set_xticks(
        range(1, len(df_fatigue.dropna(how='all', axis=1).columns)+1)[::5],)
    ax.set_xticklabels(
        range(1, len(df_fatigue.dropna(how='all', axis=1).columns)+1)[::5],
        rotation=30)
    
    fig.draw_without_rendering()
    ax.grid(alpha=0.5)
    plt.savefig(
        out_path /
        r"fatigue_all_F.png",
        bbox_inches="tight",
    )
    plt.close("all")

    print('done plotting')
    return

# %%


def plot_mvt_fish(df_cfd_2d,
                  df_x_results,
                  df_y_results,
                  df_u_fish,
                  df_v_fish,
                  bounds_poly_x,
                  bounds_poly_y,
                  ix_plot,
                  out_path,
                  xy_grid_cfd_tree,
                  nfish_plot,
                  dt_step,
                  plot_gif=False,
                  # df_u_results,
                  # df_v_results
                  ):
    """
    Plot the movement of fish in a CFD-based simulation, 
    including their swimming velocities, movement, and interactions
    with the flow.
    
        
    This function visualizes the movement of fish within a
    2D CFD grid environment. 
    For each fish:
    - It computes the fish’s movement over time and its 
    interactions with the flow field.
    - It visualizes the fish’s movement (in both swimming and flow directions) 
    on top of a background flow field.
    - The function generates plots showing the fish's trajectory, 
    flow velocities, and fish velocities at each timestep.
    - Optionally, it generates a GIF of the fish’s movement over time.
    """

    for ifish, nf in enumerate(df_x_results.index):

        if ifish <= nfish_plot:
            print(nf, len(df_x_results.index))
            df_fish_xmvt = df_x_results.loc[nf, :].dropna()
            df_fish_xy_mvt = pd.DataFrame(index=df_fish_xmvt.index,
                                          data=df_fish_xmvt.values,
                                          columns=['x'],
                                          dtype='float')
            df_fish_xy_mvt = df_fish_xy_mvt[df_fish_xy_mvt['x'] > -2.]
            df_fish_xy_mvt['y'] = df_y_results.loc[nf, :].dropna()
            # cfd neighbors
            xy_fish = np.c_[df_fish_xy_mvt.x, df_fish_xy_mvt.y]

            dist_cfd_ngbrs, ix_cfd_ngbrs = xy_grid_cfd_tree.query(
                xy_fish, k=1)

            # dx, dy
            mvt_x = df_x_results.loc[
                nf, :].diff(1).shift(-1).dropna()
            mvt_y = df_y_results.loc[
                nf, :].diff(1).shift(-1).dropna()

            df_fish_xy_mvt['dx'] = mvt_x
            df_fish_xy_mvt['dy'] = mvt_y

            # dxswim, dyswim
            df_fish_xy_mvt['dxs'] = df_u_fish.loc[
                nf, :].shift(-1).dropna()

            df_fish_xy_mvt['dys'] = df_v_fish.loc[
                nf, :].shift(-1).dropna()

            # uflow, vflow
            df_fish_xy_mvt['x_uf'] = df_cfd_2d.iloc[
                ix_cfd_ngbrs,
                np.where(df_cfd_2d.columns == 'u_f')[0]].values*dt_step
            df_fish_xy_mvt['y_vf'] = df_cfd_2d.iloc[
                ix_cfd_ngbrs,
                np.where(df_cfd_2d.columns == 'v_f')[0]].values*dt_step

            df_fish_xy_mvt = df_fish_xy_mvt.dropna()

            plt.ioff()
            fig = plt.figure(figsize=(12, 4), dpi=300)
            ax = fig.add_subplot(111)

            ax.plot(bounds_poly_x, bounds_poly_y, c="k")

            ax.quiver(df_cfd_2d.loc[:, 'xcfd_grid'].values[::ix_plot],
                      df_cfd_2d.loc[:, 'ycfd_grid'].values[::ix_plot],
                      df_cfd_2d.loc[:, 'u_f'].values[::ix_plot],
                      df_cfd_2d.loc[:, 'v_f'].values[::ix_plot],
                      color="gray",
                      scale=1,
                      width=0.01,
                      alpha=0.915,
                      headwidth=2,
                      headlength=1.5,
                      scale_units="xy",
                      units="xy",
                      angles="xy")

            ax.scatter(
                df_fish_xy_mvt.x.values.astype(float),
                df_fish_xy_mvt.y.values.astype(float),
                c='k', alpha=0.5, marker='.')

            ax.quiver(
                df_fish_xy_mvt.x.values.astype(float),
                df_fish_xy_mvt.y.values.astype(float),
                df_fish_xy_mvt.x_uf.values.astype(float),
                df_fish_xy_mvt.y_vf.values.astype(float),
                color="b",
                scale=1,
                width=0.02,
                alpha=0.5,
                headwidth=2,
                headlength=2.5,
                scale_units="xy",
                units="xy",
                angles="xy",
                label='flow')

            ax.quiver(
                df_fish_xy_mvt.x.values.astype(float),
                df_fish_xy_mvt.y.values.astype(float),
                df_fish_xy_mvt.dxs.values.astype(float),
                df_fish_xy_mvt.dys.values.astype(float),
                color="r",
                scale=1,
                width=0.02,
                alpha=0.5,
                headwidth=2,
                headlength=2.5,
                scale_units="xy",
                units="xy",
                angles="xy",
                label='swim')

            ax.quiver(
                df_fish_xy_mvt.x.values.astype(float),
                df_fish_xy_mvt.y.values.astype(float),
                df_fish_xy_mvt.dx.values.astype(float),
                df_fish_xy_mvt.dy.values.astype(float),
                color="g",
                scale=1,
                width=0.02,
                alpha=0.915,
                headwidth=2,
                headlength=2.5,
                scale_units="xy",
                units="xy",
                angles="xy",
                label='move')

            ax.set_xlim([min(bounds_poly_x)-0.1, max(bounds_poly_x)+0.1])
            ax.set_ylim([min(bounds_poly_y)-0.1, max(bounds_poly_y)+0.1])
            ax.legend(loc=0)
            ax.grid(alpha=0.2, linestyle='-.')

            plt.savefig((out_path /
                         (r'fish_id_%d.png' % nf)),
                        bbox_inches='tight')

            plt.close('all')
            if plot_gif:
                print('plotting gif, this takes some time')
                all_images = []

                for tv in tqdm.tqdm(df_fish_xy_mvt.index):
                    fig2 = plt.figure(figsize=(12, 4), dpi=200)
                    canvas = FigureCanvas(fig2)
                    ax2 = fig2.add_subplot(111)

                    ax2.plot(bounds_poly_x, bounds_poly_y, c="k")
                    ax2.quiver(
                        df_cfd_2d.loc[:, 'xcfd_grid'].values[::ix_plot],
                        df_cfd_2d.loc[:, 'ycfd_grid'].values[::ix_plot],
                        df_cfd_2d.loc[:, 'u_f'].values[::ix_plot],
                        df_cfd_2d.loc[:, 'v_f'].values[::ix_plot],
                        color="gray",
                        scale=1,
                        width=0.025,
                        alpha=0.915,
                        headwidth=2,
                        headlength=2.5,
                        scale_units="xy",
                        units="xy",
                        angles="xy")

                    ax2.plot(
                        df_fish_xy_mvt.loc[:, 'x'],
                        df_fish_xy_mvt.loc[:, 'y'], c='r', alpha=0.75)
                    ax2.scatter(
                        df_fish_xy_mvt.loc[tv, 'x'],
                        df_fish_xy_mvt.loc[tv, 'y'],
                        marker='o',
                        s=50,
                        edgecolor='c', facecolor='blue'
                    )

                    ax2.quiver(df_fish_xy_mvt.loc[tv, 'x'],
                               df_fish_xy_mvt.loc[tv, 'y'],
                               df_fish_xy_mvt.loc[tv, 'dx'],
                               df_fish_xy_mvt.loc[tv, 'dy'],
                               color="b",
                               scale=1,
                               width=0.052,
                               alpha=0.915,
                               headwidth=2,
                               headlength=3.5,
                               scale_units="xy",
                               units="xy",
                               angles="xy")

                    ax2.set_xlim([min(bounds_poly_x)-0.1, max(bounds_poly_x)+0.1])
                    ax2.set_ylim([min(bounds_poly_y)-0.1, max(bounds_poly_y)+0.1])
                    ax2.grid(alpha=0.5)
                    ax2.set_title('Time step = %0.2f s' % tv)
                    ax2.set_xlabel('X [m]')
                    ax2.set_ylabel('Y [m]')
                    canvas.draw()
                    image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
                    image = image.reshape(
                        canvas.get_width_height()[::-1] + (4,))

                    all_images.append(image)
                    plt.close(fig2)
                print('saving gif')
                imageio.mimsave(
                    (out_path /
                     (r'fish_id_%d.gif' % nf)),
                    all_images, fps=4)
                plt.close('all')

    return
