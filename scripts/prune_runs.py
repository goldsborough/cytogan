#!/usr/bin/env python3

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('runs-path')
parser.add_argument('--dry', action='store_true')
options = parser.parse_args()

for run in os.listdir(options.runs_path):
    contents = os.listdir(run)
    if not ('checkpoints' in contents or 'summaries' in contents):
        if options.dry:
            print('rm -r {0}'.format(run))
        else:
            os.rmdir(run)
