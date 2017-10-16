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
lsgan = load('lsgan')
wgan = load('wgan')
dcgan = load('dcgan')

images = []
for i in range(2):
    images += [x[i] for x in (real, lsgan, wgan, dcgan)]

figure = plot.figure(figsize=(5, 2))
for n, i in enumerate(images):
    axis = plot.subplot(2, 4, n + 1)
    axis.axis('off')
    plot.imshow(i)
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)

plot.savefig('figure.png', bbox_inches='tight', pad_inches=0)
