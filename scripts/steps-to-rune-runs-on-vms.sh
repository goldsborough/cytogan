#!/bin/bash

git clone https://github.com/goldsborough/cytogan
gsutil config
cd cytogan
sudo python scripts/prune_runs.py --dry /data1/peter/runs
sudo python scripts/prune_runs.py /data1/peter/runs
sudo scripts/copy-runs-to-gs.sh
