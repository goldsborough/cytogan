#!/usr/bin/env python3

import argparse
import os
import subprocess
import numpy as np
import scipy.misc
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', default='animated.gif')
parser.add_argument('-r', '--skip-rate', type=int, default=1)
parser.add_argument('-d', '--delay', type=int, default=10)
parser.add_argument('-a', '--annotate', nargs='+')
parser.add_argument('--frames-per-annotation', type=int, default=1)
parser.add_argument('--annotation-height', type=int)
parser.add_argument('--annotated-path')
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('files', nargs='+')
options = parser.parse_args()

options.files.sort(key=lambda p: int(os.path.basename(p).split('.')[0]))

if options.annotate is not None:
    if options.annotated_path is None:
        options.annotated_path = os.path.join(
            os.path.dirname(options.files[0]), 'annotated')
    if not os.path.exists(options.annotated_path):
        os.makedirs(options.annotated_path)

    annotations = np.repeat(options.annotate, options.frames_per_annotation)

    height, width = scipy.misc.imread(options.files[0]).shape[:2]
    print(f'Assuming dimensions {height}x{width} for images')

    if options.annotation_height is None:
        options.annotation_height = int(1 / 6 * height)

    generator = tqdm.tqdm(
        enumerate(zip(options.files, annotations)),
        unit=' images',
        desc='Annotating images')

    for n, (file, annotation) in generator:
        new_path = os.path.join(options.annotated_path, os.path.basename(file))

        command = f'convert {file} '.split()
        command += f'-size {width}x{options.annotation_height} '.split()
        command += f'-background Black -fill white -gravity south-west '.split()
        command += [f'caption:{annotation}']
        command += f'-composite {new_path}'.split()
        subprocess.run(command, check=True)

        # Replace the path with the annotated path
        options.files[n] = new_path

files = options.files[::options.skip_rate]
files = ' '.join(files)

command = 'convert -background white -alpha remove -duplicate 1,-2-1 '
command += f'-loop 0 -delay {options.delay} '
command += f'{files} {options.output}'

if options.verbose:
    print(command)
subprocess.run(command.split(), check=True)
