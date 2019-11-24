import os
import glob
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from NesrHydrusAnalyst import *

src = '../Datasets/H3D2_SandDitch0011'

print(read_selector_in(src, geom='3D'))

