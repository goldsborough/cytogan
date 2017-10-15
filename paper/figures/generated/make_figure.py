#!/usr/bin/env python3

import os

import matplotlib.pyplot as plot
import scipy.misc

def load(path):
    images = []
    for file in os.listdir(path):
        if file.endswith('.png'):
            image = scipy.misc.imread(os.path.join(path, file))
            images.append(image)
    return images

real = load('real')
fake = load('fake')
assert len(real) == len(fake), (len(real), len(fake))

figure = plot.figure(figsize=(8, 2))
for n, i in enumerate(real + fake):
    axis = plot.subplot(2, len(real), n + 1)
    plot.imshow(i)
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)

plot.savefig('figure.png')
