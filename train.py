#!/usr/bin/env python3

from cell_data import CellData

cell_data = CellData(
    metadata_file_path='../data/BBBC021_v1_image.csv',
    labels_file_path='../data/BBBC021_v1_moa.csv',
    image_root='cells',
    patterns=['Week1_22123'])
print(cell_data.next_batch_of_images(10).values()[0])
