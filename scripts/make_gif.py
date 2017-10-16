#!/usr/bin/env python3

import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', default='animated.gif')
parser.add_argument('-r', '--skip-rate', type=int, default=1)
parser.add_argument('-d', '--delay', type=int, default=10)
parser.add_argument('files', nargs='+')
options = parser.parse_args()

options.files.sort(key=lambda p: int(os.path.basename(p).split('.')[0]))
files = options.files[::options.skip_rate]
files = ' '.join(files)

command = 'convert -background white -alpha remove -duplicate 1,-2-1 '
command += f'-loop 0 -delay {options.delay} '
command += f'{files} {options.output}'

print(command)
subprocess.run(command.split())
