#!/usr/bin/env python3

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('runs')
parser.add_argument('--dry', action='store_true')
options = parser.parse_args()

print(options)

assert os.path.exists(options.runs), 'Run path does not exist!'

for run in os.listdir(options.runs):
    run_path = os.path.join(options.runs, run)
    contents = os.listdir(run_path)

    if 'checkpoints' in contents or 'summaries' in contents:
        print('Keeping {0}'.format(run_path))
    else:
        print('rm -r {0}'.format(run_path))
        if not options.dry:
            os.rmdir(run_path)
