#!/usr/bin/env python
# coding: utf-8

# # Reforming the basic functions to be non-generic

# ## Reforming the `draw_full_contour` function to simpler functions

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as LSCm
from scipy.interpolate import griddata
from scipy.signal import savgol_filter
from scipy import integrate
from NesrHydrusAnalyst import *


# In[68]:


def draw_contour(X,
                 Z,
                 M,
                 levels=None,
                 plot_title="ElNesr cross sectional contour map",
                 x_step=10.,
                 z_step=25.,
                 mirror_x=False,
                 mirror_z=False,
                 return_figure_object=False,
                 fig_size=None):
    '''

    '''
    if levels is None:
        levels = get_legend_range(np.nanmin(M), np.nanmax(M))

    fig = plt.figure(
        num=None, figsize=fig_size, dpi=80, facecolor='w', edgecolor='k')
    origin = 'lower'

    if levels is None:
        #         print(M.min(), M.max())
        try:
            # levels = get_legend_range(M.min(), M.max())
            #np.arange(0.15, 0.42, 0.03)
            # np.arange(0.15, 0.42, 0.03)
            levels = get_legend_range(np.nanmin(M), np.nanmax(M))
        except:
            levels = get_legend_range(-.15, 0.15)

    CS_lines = plt.contour(
        X,
        Z,
        M,
        levels,
        cmap=plt.cm.Accent_r,
        linewidths=(0.9, ),
        origin=origin,
        extend='both')

    CS_fill = plt.contourf(
        X, Z, M, levels, cmap=plt.cm.YlGn, origin=origin, extend='both')

    CS_fill.cmap.set_under('oldlace')
    CS_fill.cmap.set_over('darkslategrey')
    plt.title(plot_title)
    plt.ylabel("Depth (cm)")
    cols = plt.cm.Accent_r(CS_lines.norm(CS_lines.levels))
    # plt.clabel(CS_lines, linewidths=4, fmt='%2.2f', fontsize='x-large',
    plt.clabel(
        CS_lines,
        fmt='%2.2f',
        fontsize='x-large',
        colors=cols,
        inline=True,
        inline_spacing=10)
    plt.colorbar(CS_fill)

    def adjust_max_and_min(_min, _max, _step):
        nn, xx, ss = _min, _max, _step
        if xx <= 0.:
            nn, xx = xx, nn
            if ss > 0:
                ss = -ss
        return nn, xx, ss

    def adjust_axis_labels(_min, _max, _step):
        nn, xx, ss = adjust_max_and_min(_min, _max, _step)

        x_list = np.arange(nn, xx, ss)
        if abs(x_list[-1] - xx) > 3:  # The last number is far enough from
            # the maximum element
            x_list = np.hstack([x_list, xx])
        else:  # The last number is too close to the maximum element
            x_list = np.hstack([x_list[:-1], xx])
        return x_list

    def adjust_mirrored_labels(_min, _max, _step):
        #         print(_min, _max, _step)
        nn, xx, ss = adjust_max_and_min(_min, _max, _step)
        x_mid = (xx - nn) / 2.

        if x_mid < 0:
            right_list = adjust_axis_labels(min(x_mid, xx), max(x_mid, xx), ss)
            left_list = right_list - x_mid
            right_list = x_mid - right_list

            left_list.sort()

            label_list = np.hstack([left_list[:-1], right_list])

            real_list = x_mid - label_list
            label_list = label_list[::-1]
            return real_list, label_list
        elif x_mid > 0:
            right_list = adjust_axis_labels(x_mid, xx, ss)
            left_list = 2 * x_mid - right_list
            left_list.sort()
            real_list = np.hstack([left_list[:-1], right_list])
            label_list = real_list - x_mid
            label_list = tuple(['{:3.1f}'.format(x) for x in label_list])
            return real_list, label_list
        else:
            real_list, label_list = None, None
            return real_list, label_list

    if mirror_x:
        if x_step is not None:
            ticks, labels = adjust_mirrored_labels(X.min(), X.max(), x_step)
            plt.xticks(ticks, labels)
    else:  # No Mirroring
        if x_step is not None:
            plt.xticks(adjust_axis_labels(X.min(), X.max(), x_step))

    if mirror_z:
        if z_step is not None:
            ticks, labels = adjust_mirrored_labels(Z.min(), Z.max(), z_step)
            plt.yticks(ticks, labels)
    else:  # No Mirroring
        if z_step is not None:
            plt.yticks(adjust_axis_labels(Z.min(), Z.max(), z_step))
    ax = plt.gca()
    ax.grid(True, zorder=0)
    if return_figure_object:
        return fig
    else:
        plt.show()


def draw_full_contour(data_frame,
                      variable=0,
                      time_step=180,
                      grid=0.5,
                      crosses=35.,
                      tol=10.,
                      section='x',
                      levels=None,
                      plot_title="ElNesr cross sectional contour map",
                      return_arrays=True,
                      x_step=None,
                      z_step=None,
                      mirror_x=False,
                      mirror_z=False,
                      is2d=False,
                      output_the_contour=True,
                      is_axisymmetric=False,
                      return_figure_object=False,
                      fig_size=None):
    '''
    Either (1) set the return_arrays to True and use on right 
                hand side of equal sign, 
    OR     (2) set the return_arrays to False and use the function as is.
    Examples:
    (1)
       arrays = draw_full_contour(data_frame,variable, time_step, grid, 
                                   crosses, tol, section)
       It will draws the chart AND sets arrays=X, Z, M, levels
    (2)
       draw_full_contour(data_frame,variable, time_step, grid, crosses, 
                           tol, section, return_arrays=False)    
    '''
    #     print('is2d=', is2d)
    X, Z, M, x_vals, z_vals = get_grid_values(
        data_frame,
        variable,
        time_step,
        grid,
        crosses,
        tol,
        section,
        is2d=is2d)
    # print(x_vals.shape, z_vals.shape, X.shape, Z.shape, M.shape)
    if levels is None:
        levels = get_legend_range(np.nanmin(M),
                                  np.nanmax(M))  #np.arange(0.15, 0.42, 0.03)

    mn, mx = np.nanmin(M), np.nanmax(M)
    # print (mx,mn, mx-mn)
    if mx - mn < 0.000000001:
        print(
            'For the requested contour map of {}'.format(plot_title), end='. ')
        print("The map has one value only ({}), no contour map will be drawn.".
              format(mn))
        can_draw_figure = False
    else:
        can_draw_figure = True

    # Adjust a proportional figure size
    if fig_size is None:
        fig_size = get_fig_shape(df, section)

    if not output_the_contour and not return_figure_object:
        fig = None
    else:
        if can_draw_figure:
            fig = draw_contour(
                X,
                Z,
                M,
                levels,
                plot_title,
                x_step,
                z_step,
                mirror_x,
                mirror_z,
                return_figure_object,
                fig_size=fig_size)
        else:
            fig = None


#     exit()
    if return_arrays:
        if output_the_contour:
            if return_figure_object:
                return X, Z, M, levels, fig
            else:  # return_figure_object=False
                display(fig)
                # fig.show()
                return X, Z, M, levels
        else:  #output_the_contour=False
            if return_figure_object:
                return X, Z, M, levels, fig
            else:  # return_figure_object=False
                return X, Z, M, levels
    else:  #return_arrays=False
        if output_the_contour:
            if return_figure_object:
                return fig
            else:  # return_figure_object=False
                display(fig)
                # fig.show()
        else:  #output_the_contour=False
            if return_figure_object:
                return fig



display = print

# In[4]:


# src = '../Datasets/sample3d'
# data_frame = read_hydrus_data(folder=src, save_to_csv=False)
# v = 0
# X, Z, M, x_vals, z_vals = get_grid_values(
#     data_frame,
#     variable=v,
#     time_step=180,
#     grid=0.5,
#     crosses=35.,
#     tol=10.,
#     section='x',
#     testing=False,
#     is2d=False)


# In[5]:


# print('x_vals{}, z_vals{}, X{}, Z{}, M{}'.format(x_vals.shape, z_vals.shape,
#                                                  X.shape, Z.shape, M.shape))
# display(
#     get_available_timesteps(data_frame), get_full_dimensions(data_frame),
#     get_legend_range(M.min(), M.max()), get_fig_shape(data_frame, 'y'),
#     get_fig_shape(data_frame, 'x'))
#
#
# # In[7]:


# # testing draw_full_contour function
# # data_frame_test = read_hydrus_data(save_to_csv=False)
# variable = 0  # Theta
#
# time_step = 180
# grid = 0.5  # cm
# tol = 10.
#
# section = 'y'
# crosses = 50.  #cm0)]
# _ = draw_full_contour(
#     data_frame,
#     variable,
#     time_step,
#     grid,
#     crosses,
#     tol,
#     section,
#     return_arrays=False,
#     x_step=12,
#     z_step=10,
#     mirror_x=True,
#     mirror_z=False,
#     fig_size=get_fig_shape(data_frame, section))
#
# section = 'x'
# crosses = 35.  #cm0)]
# _ = draw_full_contour(
#     data_frame,
#     variable,
#     time_step,
#     grid,
#     crosses,
#     tol,
#     section,
#     return_arrays=False,
#     x_step=12,
#     z_step=10,
#     mirror_x=True,
#     mirror_z=False,
#     fig_size=get_fig_shape(data_frame, section))
#
# # Get other data from 3D simulation
# get_one_line_df(src, simulation_name="Sand Ditch simulation", dims='3d').T
#
#
#

# In[8]:
src = '../Datasets/H3D2_SandDitch0011'
df = read_hydrus_data(folder=src, save_to_csv=False, read_velocities=True)
display(df.sample(3), get_full_simulation_info(df))


# In[9]:


df_rotated = rotate_back(df, 2.2899, rotation_axis='y')
display(get_full_simulation_info(df_rotated))


# In[34]:


# time_step = 60
# grid = 0.5  # cm
# # crosses = 15.  #cm0)]
# tol = 10.
# v_mask = {0: 'Th', 1: 'H', 2.1: 'V1', 2.2: 'V2', 2.3: 'V3'}
# v_mask_cordinates = {0: 'Th', 1: 'H', 2.1: 'Vx', 2.2: 'Vy', 2.3: 'Vz'}
# cros_mask = {'x':25.0, 'y':10.5, 'z':10.5}
# for section in ['y', 'x', 'z']:
#     crosses = cros_mask[section]
#     for variable in [0, 1, 2.1, 2.2, 2.3]:
#         display(
#             f"For variable '{v_mask_cordinates[variable]}', the simulation's "
#             f"crosssection at '{crosses} cm' of '{section}' direction is:")
#         draw_full_contour(
#             df_rotated,
#             variable,
#             time_step,
#             grid,
#             crosses,
#             tol,
#             section,
#             return_arrays=False,
#             x_step=12,
#             z_step=10,
#             mirror_x=True,
#             mirror_z=False)#,
#             fig_size=get_fig_shape(df, section))



# In[49]:


# section = 'x'
# storage = {}
# crosses = 50.
# time_step = 180
# v_mask_cordinates = {0: 'Moisture', 1: 'Head', 2.1: 'Vx', 2.2: 'Vy', 2.3: 'Vz'}
# grid=0.5

# for var in [0, 1, 2.1, 2.2, 2.3]:
#     storage[var] = draw_full_contour(
#         df_rotated,
#         variable=var,
#         time_step=time_step,
#         grid=grid,
#         crosses=crosses,
#         tol=10.,
#         section=section,
#         levels=None,
#         plot_title=f"A {crosses} cm {v_mask_cordinates[var]} cross-section "
#         f"in {section} direction at {time_step} minute",
#         return_arrays=True,
#         x_step=None,
#         z_step=None,
#         mirror_x=False,
#         mirror_z=False,
#         is2d=False,
#         output_the_contour=True,
#         is_axisymmetric=False,
#         return_figure_object=False)  #,
#                       fig_size=get_fig_shape(df_rotated, section))

#
# # In[50]:
#
#
# # Moisture values
# Y, Z, M, Lm = storage[0]
# # Velocity in X direction values
# _, _, Vx, Lv = storage[2.1]
#
#
# # In[57]:
#
#
# Lm, Lv
#
#
# # In[58]:
#
#
# [x.shape for x in (Y, Z, M, Vx)]
#
#
# # In[62]:
#
#
# MV = M*Vx
# MV.min(), MV.max()
#
#
# # In[63]:
#
#
# MV
#
#
# # In[69]:
#
#
# Lmv = get_legend_range(np.nanmin(MV), np.nanmax(MV))
# draw_contour(Y, Z, MV, Lmv)
#
#
# # In[86]:
#
#
# # THis is the full contour
# # But the discharge occur only in the bottom 2 cm
# # the grid is 0.5 cm * 0.5 cm
# # then let's take the last 4+1 points
#
# # section = 'x'  # Commented not to make conflict by the above variables
# y_length = get_full_dimensions(df_rotated)['y']
# y_grids = Y.shape[0] - 1
# # grid = 0.5  # Commented not to make conflict by the above variables
#
# d = 2.0  # the length of the discharge region
#
# ng = int(d / grid) + 1  # Number of grid points
# ng
#
#
# # In[78]:
#
#
# # The cropped arrays
#
# Mc, Vc = M[-ng:, :], Vx[-ng:, :]
#
# # The product
# MVc = Mc*Vc
# MVc
#
#
# # In[94]:
#
#
# MVc.mean(), np.abs(MVc).mean()
#
#
# # In[87]:
#
#
# # Area of the drain = 2*20 = 40 cm squared
# A = d * (y_length[1]-y_length[0])
# y_length,A
#
#
# # In[95]:
#
#
# # The drainage volume =
# vol = A * np.abs(MVc).mean()
# vol
#
#
# # In[149]:
#

time_steps = get_available_timesteps(df_rotated)
section = 'x'
crosses = 50.
v_mask_cordinates = {0: 'Moisture', 1: 'Head', 2.1: 'Vx', 2.2: 'Vy', 2.3: 'Vz'}
grid=0.5
time_storage={}
for time_step in time_steps:
    storage = {}
    for var in [0, 2.1]:#[0, 1, 2.1, 2.2, 2.3]:
        storage[var] = draw_full_contour(
            df_rotated,
            variable=var,
            time_step=time_step,
            grid=grid,
            crosses=crosses,
            tol=10.,
            section=section,
            levels=None,
            plot_title=f"A {crosses} cm {v_mask_cordinates[var]} cross-section "
            f"in {section} direction at {time_step} minute",
            return_arrays=True,
            x_step=None,
            z_step=None,
            mirror_x=False,
            mirror_z=False,
            is2d=False,
            output_the_contour=False,
            is_axisymmetric=False,
            return_figure_object=False)

    # Moisture values
    Y, Z, M, Lm = storage[0]
    # Velocity in X direction values
    _, _, Vx, Lv = storage[2.1]

    # section = 'x'  # Commented not to make conflict by the above variables
    y_length = get_full_dimensions(df_rotated)['y']
    y_grids = Y.shape[0] - 1
    # grid = 0.5  # Commented not to make conflict by the above variables

    d = 2.0  # the length of the discharge region
    region_end = 20
    region_start = region_end - d
    
    

    ng = int(d / grid) + 1  # Number of grid points

    # The cropped arrays
    Mc, Vc = M[-ng:, :], Vx[-ng:, :]

    # The product
    MVc = Mc*Vc
    
    # print(d, region_end, region_start, ng)
    
    # Area of the drain = 2*20 = 40 cm squared
    A = d * (y_length[1]-y_length[0])

    # The drainage volume = 
    vol = A * np.abs(MVc).mean()
    
    time_storage[time_step] = vol

print(time_storage)


# ## A function to calculate volumes flow in all times AnyWhere

# In[115]:


# d = 2.0  # the length of the discharge region
# region_end = 20
# region_start = region_end - d
# ng = int(d / grid) + 1  # Number of grid points
#
# display(d, region_start, region_end, ng,
#         M[-ng:, 1:8],
#         int(region_start / grid) + 2,
#         int(region_end / grid) + 3,
#         M[int(region_start / grid)+2 :int(region_end / grid) + 3, 1:8],
#        M[:,1:8])


# In[118]:


# sample_array = np.random.random(54).reshape((9, 6))
# sample_array
#
#
# # In[121]:
#
#
# sample_array[0:3,:], sample_array[6:9,:]
#
#
# # for this `sample_array` it is a **4.0cm** height and **2.5cm** width with **0.5cm** grid
#
# # In[131]:
#
#
# # for this sample_array it is a 4cm height and 2.5cm width with 0.5cm grid
#
# d = 1.0  # the length of the discharge region
#
# for rgn_e in [1.0, 2.5, 4.0]:  # region_end
#     rgn_s = rgn_e - d  # region_start
#     # ng = int(d / grid) + 1  # Number of grid points
#
#     grd_s, grd_e = int(rgn_s / grid), int(rgn_e / grid + 1)
#
#     print(d, rgn_s, rgn_e, ng, grd_s, grd_e)
#     display(sample_array[grd_s:grd_e, :], )
#

# In[148]:


def get_window_time_volumes(
        df,
        crosses,
        region_length,
        region_location,
        location_is_start=False,  #the region_location =  region_end
        section='x',
        grid=0.5,
        absolute_velocities=True
):
    time_steps = get_available_timesteps(df)
    section = section
    # crosses = 50.
    v_mask_cordinates = {
        0: 'Moisture',
        1: 'Head',
        2.1: 'Vx',
        2.2: 'Vy',
        2.3: 'Vz'
    }
    plot_title=''
#     plot_title=f"A {crosses} cm {v_mask_cordinates[var]} cross-section "\
#                 f"in {section} direction at {time_step} minute"
    grid = grid
    time_storage = {}
    velocity={'x': 2.1, 'y': 2.2, 'z': 2.3}[section.lower()]
    variables = [0, velocity]
    for time_step in time_steps:
        storage = {}
        for var in variables:  #[0, 1, 2.1, 2.2, 2.3]:
            storage[var] = draw_full_contour(
                df,
                variable=var,
                time_step=time_step,
                grid=grid,
                crosses=crosses,
                tol=10.,
                section=section,
                levels=None,
                plot_title=plot_title,
                return_arrays=True,
                x_step=None,
                z_step=None,
                mirror_x=False,
                mirror_z=False,
                is2d=False,
                output_the_contour=False,
                is_axisymmetric=False,
                return_figure_object=False)

        # Moisture values
        Y, Z, M, Lm = storage[0]
        # Velocity in X direction values
        _, _, Vx, Lv = storage[velocity]  # to get storage[2.1 if 'x']

        # section = 'x'  # Commented not to make conflict by the above variables
        y_length = get_full_dimensions(df)[{
            'x': 'y',
            'y': 'x',
            'z': 'y'
        }[section.lower()]]  # to get_full_dimensions(df)['y'] if 'x'

        y_grids = Y.shape[0] - 1

        if location_is_start:
            region_start = region_location
            region_end = region_start + region_length
        else:
            region_end = region_location
            region_start = region_end - region_length


#         ng = int(dregion_length / grid) + 1  # Number of grid points
        grd_s, grd_e = int(region_start / grid), int(region_end / grid + 1)

        # The cropped arrays
        Mc, Vc = M[grd_s:grd_e, :], Vx[grd_s:grd_e, :]

        # The product
        MVc = Mc * Vc

        # Area of the drain = 2*20 = 40 cm squared
        A = region_length * (y_length[1] - y_length[0])

        # The drainage volume =
        if absolute_velocities:
            vol = A * np.nanmean(np.abs(MVc))
        else:
            vol = A * p.nanmean(MVc)

        time_storage[time_step] = vol
        
    return time_storage


# In[147]:


# Testing
print(get_window_time_volumes(
    df_rotated,
    50,
    2,
    region_location=21,
    location_is_start=False,  #the region_location =  region_end
    section='x',
    grid=0.5,
    absolute_velocities=True))


# In[ ]:




