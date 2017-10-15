#!/usr/bin/env python3

import argparse
import os.path

import matplotlib.pyplot as plot
import scipy.misc

parser = argparse.ArgumentParser()
parser.add_argument('folder')
options = parser.parse_args()


def load(dirname, basename):
    path = os.path.join(dirname, '{0}.png'.format(basename))
    return scipy.misc.imread(path)


lhs = load(options.folder, 'lhs')
rhs = load(options.folder, 'rhs')
base = load(options.folder, 'base')
result = load(options.folder, 'result')

figure = plot.figure(figsize=(10, 2))

axis = plot.subplot(141)
axis.get_xaxis().set_visible(False)
axis.get_yaxis().set_visible(False)
plot.imshow(lhs)

axis.text(
    1.22,
    0.5,
    r'$-$',
    horizontalalignment='center',
    verticalalignment='center',
    transform=axis.transAxes,
    fontsize=30)

axis = plot.subplot(142)
axis.get_xaxis().set_visible(False)
axis.get_yaxis().set_visible(False)
plot.imshow(rhs)

axis.text(
    1.22,
    0.5,
    r'$+$',
    horizontalalignment='center',
    verticalalignment='center',
    transform=axis.transAxes,
    fontsize=30)

axis = plot.subplot(143)
axis.get_xaxis().set_visible(False)
axis.get_yaxis().set_visible(False)
plot.imshow(base)

axis.text(
    1.22,
    0.5,
    r'$=$',
    horizontalalignment='center',
    verticalalignment='center',
    transform=axis.transAxes,
    fontsize=30)

axis = plot.subplot(144)
axis.get_xaxis().set_visible(False)
axis.get_yaxis().set_visible(False)
plot.imshow(result)

plot.tight_layout()
plot.savefig('figure-{0}.png'.format(options.folder))
