import os
import glob
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as LSCm
from scipy.interpolate import griddata
from scipy.signal import savgol_filter
from scipy import integrate

from NesrHydrusAnalyst import *
# src = '../Datasets/sample3d'
src = '../Datasets/H3D2_SandDitch0011'
data_frame= read_hydrus_data(folder=src, save_to_csv=False)
print(data_frame.shape)
v=0
X, Z, M, x_vals, z_vals = get_grid_values(data_frame,variable=v)
print(get_legend_range(np.nanmin(M), np.nanmax(M)),
      '\nSahpes:\nx_vals=> {}\nz_vals=> {}\nX=> {}\nZ=> {}\nM=> {}'.format(
          x_vals.shape, z_vals.shape,X.shape, Z.shape, M.shape))