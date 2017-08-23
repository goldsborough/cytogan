#!/usr/bin/env python3

import numpy as np

from cytogan.cell_data import CellData
from cytogan.score import score_profiles

cell_data = CellData(
    metadata_file_path='../data/BBBC021_v1_image.csv',
    labels_file_path='../data/BBBC021_v1_moa.csv',
    image_root='data/cells',
    patterns=['Week4_27481/G02', 'Week4_27521/B05'])

b = cell_data.all_images()
k = list(b.keys())
profile = np.arange(100)
p = {}
for i in k:
    if cell_data.metadata.loc[i]['compound'] == 'anisomycin':
        p[i] = np.arange(100)
    else:
        p[i] = -np.arange(100)

d = cell_data.create_dataset_from_profiles(p)
c, a = score_profiles(d)

print(c)
print(a)
