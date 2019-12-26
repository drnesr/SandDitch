import os
import glob
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from NesrHydrusAnalyst import *

sources = [f'../Datasets/H3D2_SandDitch0014{x}' for x in ('', 'b', 'c', 'd', 'e')]
for source in sources:
    print(f'\nFor the {source.split("/")[2][-3:]} simulation')
    export_all_csvs(source, rotation_angle=0)