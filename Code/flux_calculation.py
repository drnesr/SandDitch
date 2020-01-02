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
crossing = 45
flow = get_window_time_volumes(dfs[0], crossing, 3,28)
print(crossing, flow)