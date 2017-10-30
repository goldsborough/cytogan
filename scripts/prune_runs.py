#!/usr/bin/env python3

import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('runs')
parser.add_argument('--dry', action='store_true')
parser.add_argument('--keep-with-figures', action='store_true')
parser.add_argument('--removed-only', action='store_true')
options = parser.parse_args()

assert os.path.exists(options.runs), 'Run path does not exist!'

for run in os.listdir(options.runs):
    run_path = os.path.join(options.runs, run)
    contents = os.listdir(run_path)

    with_figures = options.keep_with_figures and 'figures' in contents
    if 'checkpoints' in contents or with_figures:
        if not options.removed_only:
            print('Keeping {0}'.format(run_path))
    else:
        if options.removed_only:
            print(run_path)
        else:
            print('rm -r {0}'.format(run_path))
        if not options.dry:
            shutil.rmtree(run_path)
