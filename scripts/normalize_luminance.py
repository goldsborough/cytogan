#!/usr/bin/env python3

import argparse
import os

import numpy as np
import scipy.misc
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--images', nargs='+')
parser.add_argument('-f', '--folder')
parser.add_argument('-o', '--out', default='normalized')
options = parser.parse_args()

assert options.images or options.folder

if options.folder:
    options.images = os.listdir(options.folder)

if not os.path.exists(options.out):
    os.makedirs(options.out)

def normalize_luminance(image):
    maxima = image.max(axis=(0, 1))
    if maxima.sum() > 0:
        image /= maxima.reshape(1, 1, -1)
    return image

try:
    for path in tqdm.tqdm(options.images):
        if options.folder:
            path = os.path.join(options.folder, path)
        try:
            image = scipy.misc.imread(path).astype(np.float32) / 255.0
        except:
            continue
        normalized = normalize_luminance(image)
        out_path = os.path.join(options.out, os.path.basename(path))
        scipy.misc.imsave(out_path, normalized)
except KeyboardInterrupt:
    pass
