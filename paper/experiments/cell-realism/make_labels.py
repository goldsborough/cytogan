#!/usr/bin/env python3

import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--real', default='real')
parser.add_argument('-f', '--fake', default='fake')
parser.add_argument('-p', '--prefix')
options = parser.parse_args()

assert os.path.exists(options.real)
assert os.path.exists(options.fake)

def get_files(path):
    for file in os.listdir(path):
        if file.endswith('.png'):
            yield os.path.join(options.prefix, path, file)

real_files = get_files(options.real)
fake_files = get_files(options.fake)

real_map = {file: True for file in real_files}
fake_map = {file: False for file in fake_files}
labels = {**real_map, **fake_map}

print(json.dumps(labels, indent=4))
