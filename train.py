#!/usr/bin/env python3

import numpy as np

from cell_data import CellData
from score import score_profiles

cell_data = CellData(
    metadata_file_path='../data/BBBC021_v1_image.csv',
    labels_file_path='../data/BBBC021_v1_moa.csv',
    image_root='data/cells',
    patterns=['Week4_27481'])

b = cell_data.all_images()
k = list(b.keys())
profile = np.arange(100)
p = {i: profile for i in k}

d = cell_data.create_dataset_from_profiles(p)
print(d)
c, a = score_profiles(d)

print(c)
print(a)
