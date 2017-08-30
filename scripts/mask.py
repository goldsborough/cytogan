#!/usr/bin/env python3

import argparse
import glob
import multiprocessing
import os
import os.path
import signal
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
    key = plate + os.sep + filename
    labels = key.str.contains(regex_pattern)
    return metadata.loc[labels]


def parse_paths(metadata_path, patterns, mask_root_path, image_root_path):
    metadata = pd.read_csv(metadata_path)
    if patterns:
        metadata = filter_metadata(metadata, patterns)

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

        yield ImagePath(dna_image_path, tubulin_image_path, actin_image_path,
                        mask_path, mask_name)


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
    mask_min_row = minima.row[mask_index] + 1
    mask_max_row = maxima.row[mask_index] + 1
    mask_min_column = minima.column[mask_index]
    mask_max_column = maxima.column[mask_index]
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

    index = pd.MultiIndex.from_arrays([mask.flatten()], names=['label'])

    row_series = pd.Series(row, index=index, name='row')
    col_series = pd.Series(col, index=index, name='column')
    image_series = pd.Series(image.flatten(), index=index, name='image')
    columns = pd.concat([row_series, col_series, image_series], axis=1)

    # Aggregate over the pixel values of the mask.
    grouped = columns.groupby(level=0)
    return grouped.min(), grouped.max()


def clip_crop_slices(slices, clip):
    output = list(slices)
    for i, s in enumerate(slices):
        if s.stop - s.start > clip:
            output[i] = slice(s.start, s.start + clip)
    return output


def process_channel(image, mask, output_size):
    minima, maxima = get_mask_boundaries(image, mask)
    # The first mask is the entire segmentation
    for mask_index in minima.index.values:
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

    cells = []
    for dna, actin, tubulin in zip(dna_masked, actin_masked, tubulin_masked):
        cell = np.dstack([dna, tubulin, actin]).astype(np.uint8)
        assert np.ndim(cell) == 3
        if display:
            display_cell(dna, tubulin, actin, cell)
        cells.append(cell)

    return cells


def save_single_cell(output_directory, image_prefix, index, image):
    filename = '{0}-{1}.png'.format(image_prefix, index)
    output_path = os.path.join(output_directory, filename)
    most_specific_directory = os.path.dirname(output_path)
    if not os.path.exists(most_specific_directory):
        os.makedirs(most_specific_directory)
        print('Creating {0}'.format(most_specific_directory))
    assert np.ndim(image) == 3
    scipy.misc.imsave(output_path, image)


class MaskJob(object):
    def __init__(self, options):
        self.options = options
        self.images_processed = 0
        self.cells_processed = 0
        self.error_count = 0
        self.cell_counts = {}

    def __call__(self, image_index, image_path):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        image_key = image_path.prefix
        image = read_images(image_path)
        try:
            cells = process_image(image, self.options.size,
                                  self.options.display)
        except Exception as error:
            raise RuntimeError(image_key, error)
        return image_key, cells

    def on_success(self, args):
        image_key, cells = args
        # Count the difference so we know when cells_processed is > limit
        cells_at_start = self.cells_processed
        for cell_index, cell in enumerate(cells):
            assert np.ndim(cell) == 3
            assert cell.shape[2] == 3
            if not self.options.display:
                save_single_cell(self.options.output, image_key, cell_index,
                                 cell)
            self.cells_processed += 1
            if self.cells_processed == self.options.cell_limit:
                break
        count = self.cells_processed - cells_at_start
        print('Generated {0:>3} cells for {1} ...'.format(count, image_key))
        self.cell_counts[image_key] = count
        self.images_processed += 1

    def on_error(self, error):
        image_key, real_error = error.args
        self.error_count += 1
        print('Failed to process {0}: {1}'.format(image_key, repr(real_error)))


def mask_images(image_paths, options):
    job = MaskJob(options)
    pool = multiprocessing.Pool()
    try:
        for image_index, image_path in enumerate(image_paths):
            if job.cells_processed == options.cell_limit or \
               job.images_processed == options.image_limit:
                break
            pool.apply_async(
                job, [image_index, image_path],
                callback=job.on_success,
                error_callback=job.on_error)
    except KeyboardInterrupt:
        print()
    pool.close()
    pool.join()

    return job.images_processed, job.cells_processed, \
           job.cell_counts, job.error_count


def parse():
    parser = argparse.ArgumentParser(description='Generate masked cell images')
    parser.add_argument('-p', '--pattern', action='append')
    parser.add_argument('-i', '--image-path', default='.')
    parser.add_argument('-o', '--output', default='.')
    parser.add_argument('-d', '--metadata', required=True)
    parser.add_argument('-s', '--size', type=int, default=128)
    parser.add_argument('-m', '--masks', required=True)
    parser.add_argument('--cell-limit', type=int)
    parser.add_argument('--cell-count-csv')
    parser.add_argument('--image-limit', type=int)
    parser.add_argument('--display', action='store_true')
    return parser.parse_args()


def main():
    options = parse()
    if options.cell_limit == 0 or options.image_limit == 0:
        return
    image_paths = parse_paths(options.metadata, options.pattern, options.masks,
                              options.image_path)

    start = time.time()
    stats = mask_images(image_paths, options)
    images_processed, cells_processed, cell_counts, error_counts = stats
    elapsed = time.time() - start
    print('Processed {0:,} images into {1:,} cells in {2:.2f}s ({3} errors)'.
          format(images_processed, cells_processed, elapsed, error_counts))

    if options.cell_count_csv:
        with open(options.cell_count_csv, 'w') as file:
            file.write('key, number_of_cells\n')
            lines = ['{0},{1}\n'.format(k, c) for k, c in cell_counts.items()]
            file.writelines(lines)


if __name__ == '__main__':
    main()
