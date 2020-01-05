from NesrHydrusAnalyst import *
import pandas as pd
import numpy as np

extensions = ('', 'a', 'b', 'c', 'd', 'e')
sources = [f'../Datasets/H3D2_SandDitch0014{x}' for x in extensions]

dfs = [
    pd.read_csv(os.path.join(sources[x], 'Nesr', '1-Original_Grid.csv'))
    for x in range(len(extensions))
]
# # Test at different cross sections
# for crossing in [0, 10, 20, 30, 45]:
#     flow = get_uneven_spans_area(
#         get_window_time_volumes(dfs[0], crossing, 2, 21))
#     print(
#         f'For the cross section at {crossing:5.2f} cm, The total passing flow = '
#         f'{flow:6.4f} qcm')

# for crossing in [0, 10, 20, 30, 45]:
#     flow = get_window_time_volumes(dfs[0], crossing, 2, 21)
#     print(crossing, flow)

# Through X direction, and the window is is Y and Z directions (all Y and section of Z)
# crossing = 45
# flow = get_window_time_volumes(dfs[0], crossing, 3,28, section='x')
# print(crossing, flow)

# # Take the last 3 cm of each setion
# data = (('x', 'z', 45, 3, 28), ('x', 'y', 45, 3, 30),
#         ('y', 'z', 30, 3, 28), ('y', 'x', 30, 3, 45),
#         ('z', 'x', 28, 3, 45), ('z', 'y', 28, 3, 30), )
# for case in data:
#     sec, prt, crossing, s_length, s_end = case
#     flow = get_window_time_volumes(dfs[0], crossing, s_length, s_end, section=sec, partition_axis=prt)
#     print(case, flow)

#
# # Take the most important sections
# data = (('Runoff', 'x', 'z', 45, 3, 28),
#         ('Drainage', 'x', 'z', 45, 2, 2),
#         ('Flux', 'z', 'x', 28, 3, 45),
#         ('Flux0', 'z', 'x', 28, 3, 3),
#         ('Evaporation', 'z', 'x', 28, 42, 42),
#         )
#
# '''
# First results
# ('Runoff', 'x', 'z', 45, 3, 28) 1.7358156406026053
# ('Drainage', 'x', 'z', 45, 2, 2) 0.01640582312869202
# ('Flux', 'z', 'x', 28, 3, 45) 0.
#
# '''
#
# for fltr_neg in (True, False):
#     negs = 'Negatives are excluded' if fltr_neg else 'Negatives are included'
#     for abs_vel in (True, False):
#         abss = 'Absolute values were taken' if abs_vel else 'Values were taken with sign'
#         for case in data:
#             caption, sec, prt, crossing, s_length, s_end = case
#             flow = get_uneven_spans_area(get_window_time_volumes(dfs[0], crossing, s_length, s_end,
#                                                                  section=sec, partition_axis=prt,
#                                                                  filter_negatives=fltr_neg,
#                                                                  absolute_velocities=abs_vel))
#             print(negs, abss, case, flow)


# Testing boundary fluxes

# print(read_boundary_data(sources[0], 19, 22))

export_all_csvs(sources[0])