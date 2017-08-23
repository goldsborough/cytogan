#!/usr/bin/env python3

import argparse
import glob
import os.path
import time
from collections import namedtuple

import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import scipy.misc

ImagePath = namedtuple('ImagePath', 'dna, tubulin, actin, mask, prefix')
Image = namedtuple('Image', 'dna, tubulin, actin, mask')


def filter_metadata(metadata, patterns):
    regex_pattern = '|'.join(patterns)
    plate = metadata['Image_Metadata_Plate_DAPI']
    filename = metadata['Image_FileName_DAPI']
    key = plate + '/' + filename
    labels = key.str.contains(regex_pattern)
    return metadata.loc[labels]


def parse_paths(metadata_path, patterns, mask_root_path, image_root_path):
    metadata = pd.read_csv(metadata_path)
    if patterns:
        metadata = filter_metadata(metadata, patterns)

    images_paths = []
    for index, row in metadata.iterrows():
        plate = row['Image_Metadata_Plate_DAPI']
        image_path_prefix = os.path.join(image_root_path, plate)

        dna_image_path = os.path.join(image_path_prefix,
                                      row['Image_FileName_DAPI'])
        actin_image_path = os.path.join(image_path_prefix,
                                        row['Image_FileName_Actin'])
        tubulin_image_path = os.path.join(image_path_prefix,
                                          row['Image_FileName_Tubulin'])

        mask_name = os.path.join(
            plate, os.path.splitext(row['Image_FileName_DAPI'])[0])
        mask_glob = os.path.join(mask_root_path,
                                 '{0}_Cell.*'.format(mask_name))
        glob_result = glob.glob(mask_glob)
        if not glob_result:
            continue
        mask_path = glob_result[0]

        image_path = ImagePath(dna_image_path, tubulin_image_path,
                               actin_image_path, mask_path, mask_name)
        images_paths.append(image_path)

    return images_paths


def load_image(path):
    image = scipy.misc.imread(path, 'L')
    return image / image.max() * 255.0


def read_images(image_path):
    dna_image = load_image(image_path.dna)
    tubulin_image = load_image(image_path.tubulin)
    actin_image = load_image(image_path.actin)
    mask_image = scipy.misc.imread(image_path.mask, 'L').astype(np.uint8)
    return Image(dna_image, tubulin_image, actin_image, mask_image)


def get_crop_slices(minima, maxima, mask_index):
    mask_min_row = minima['row'][mask_index] + 1
    mask_max_row = maxima['row'][mask_index] + 1
    mask_min_column = minima['column'][mask_index]
    mask_max_column = maxima['column'][mask_index]
    row_slice = slice(mask_min_row, mask_max_row)
    column_slice = slice(mask_min_column, mask_max_column)

    return row_slice, column_slice


def crop_channel(row_slice, column_slice, image, mask, output_size):
    width = row_slice.stop - row_slice.start
    height = column_slice.stop - column_slice.start

    row_start = max(0, (output_size - width) // 2)
    row_end = row_start + width
    column_start = max(0, (output_size - height) // 2)
    column_end = column_start + height

    cropped_image = np.zeros((output_size, output_size))
    image_crop = image[row_slice, column_slice]
    cropped_image[row_start:row_end, column_start:column_end] = image_crop

    cropped_mask = np.zeros_like(cropped_image)
    mask_crop = mask[row_slice, column_slice]
    cropped_mask[row_start:row_end, column_start:column_end] = mask_crop

    return cropped_image, cropped_mask


def get_mask_boundaries(image, mask):
    row, col = np.indices(mask.shape)
    row, col = row.flatten(), col.flatten()

    mask_vector = mask.flatten()
    image_vector = image.flatten()

    index = pd.MultiIndex.from_arrays([mask_vector], names=['label'])

    row_series = pd.Series(row, index=index, name='row')
    col_series = pd.Series(col, index=index, name='column')
    image_series = pd.Series(image_vector, index=index, name='image')
    mask_series = pd.Series(mask_vector, index=index, name='mask')

    columns = pd.concat(
        [row_series, col_series, image_series, mask_series], axis=1)
    columns = columns.groupby(level=0)

    maxima = columns.aggregate(np.max)
    minima = columns.aggregate(np.min)

    return minima, maxima


def clip_crop_slices(slices, clip):
    output = list(slices)
    for i, s in enumerate(slices):
        if s.stop - s.start > clip:
            output[i] = slice(s.start, s.start + clip)
    return output


def process_channel(image, mask, output_size):
    minima, maxima = get_mask_boundaries(image, mask)

    # The first mask is the entire segmentation
    for mask_index in range(1, min(len(minima), len(maxima))):
        slices = get_crop_slices(minima, maxima, mask_index)
        row_slice, column_slice = clip_crop_slices(slices, output_size)
        crops = crop_channel(row_slice, column_slice, image, mask, output_size)
        cropped_image, cropped_mask = crops

        masked_image = np.zeros_like(cropped_image)
        mask_indices = np.where(cropped_mask == mask_index)
        masked_image[mask_indices] = cropped_image[mask_indices]

        yield masked_image


def display_cell(dna, tubulin, actin, cell):
    plot.figure(figsize=(20, 3))
    for index, image in enumerate([cell, dna, actin, tubulin]):
        axis = plot.subplot(1, 4, 1 + index)
        mode = 'gray' if index > 0 else None
        plot.imshow(image, mode)
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
    plot.show()


def process_image(image, output_size, display):
    dna_masked = process_channel(image.dna, image.mask, output_size)
    actin_masked = process_channel(image.actin, image.mask, output_size)
    tubulin_masked = process_channel(image.tubulin, image.mask, output_size)
    for dna, actin, tubulin in zip(dna_masked, actin_masked, tubulin_masked):
        cell = np.dstack([dna, tubulin, actin]).astype(np.uint8)
        if display:
            display_cell(dna, tubulin, actin, cell)
        yield cell


def save_single_cell(output_directory, image_prefix, index, image):
    filename = '{0}-{1}.png'.format(image_prefix, index)
    output_path = os.path.join(output_directory, filename)
    most_specific_directory = os.path.dirname(output_path)
    if not os.path.exists(most_specific_directory):
        os.makedirs(most_specific_directory)
        print('Creating {0}'.format(most_specific_directory))
    scipy.misc.imsave(output_path, image)


def mask_images(image_paths, args):
    images_processed = 0
    cells_processed = 0
    try:
        for image_index, image_path in enumerate(image_paths):
            image = read_images(image_path)
            try:
                cells = process_image(image, args.size, args.display)
                cells_at_start = cells_processed
                for cell_index, cell in enumerate(cells):
                    if not args.display:
                        save_single_cell(args.output, image_path.prefix,
                                         cell_index, cell)
                    cells_processed += 1
                    if cells_processed == args.cell_limit:
                        return images_processed, cells_processed
                print('Generated {0:>2} cells for {1} ...'.format(
                    cells_processed - cells_at_start, image_path.prefix))
                images_processed += 1
                if images_processed == args.image_limit:
                    break
            except Exception as error:
                print('Failed to process {0}: {1}'.format(
                    image_path.prefix, repr(error)))
    except KeyboardInterrupt:
        print()

    return images_processed, cells_processed


def parse():
    parser = argparse.ArgumentParser(description='Generate masked cell images')
    parser.add_argument('-p', '--pattern', action='append')
    parser.add_argument('-i', '--image-path', default='.')
    parser.add_argument('-o', '--output', default='.')
    parser.add_argument('-d', '--metadata', required=True)
    parser.add_argument('-s', '--size', type=int, default=128)
    parser.add_argument('-m', '--masks', required=True)
    parser.add_argument('--cell-limit', type=int)
    parser.add_argument('--image-limit', type=int)
    parser.add_argument('--display', action='store_true')
    return parser.parse_args()


def main():
    args = parse()
    if args.cell_limit == 0 or args.image_limit == 0:
        return
    image_paths = parse_paths(args.metadata, args.pattern, args.masks,
                              args.image_path)

    start = time.time()
    images_processed, cells_processed = mask_images(image_paths, args)
    elapsed = time.time() - start
    print('Processed {0} images into {1} cells in {2:.2f}s'.format(
        images_processed, cells_processed, elapsed))


if __name__ == '__main__':
    main()
