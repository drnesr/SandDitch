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
src = '../Datasets/sample3d'
data_frame= read_hydrus_data(folder=src, save_to_csv=False)

v=0
X, Z, M, x_vals, z_vals = get_grid_values(data_frame,variable=v)