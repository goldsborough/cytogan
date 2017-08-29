#!/bin/bash

python3 -m cProfile -s time                             \
  cytogan/train/bbbc021.py                              \
  --epochs  20                                          \
  --model vae                                           \
  --lr 0.001                                            \
  --lr-decay 0.999                                      \
  --checkpoint-dir /data1/peter/runs/checkpoints        \
  --checkpoint-freq '1min'                              \
  --summary-dir /data1/peter/runs/summaries             \
  --summary-freq '1min'                                 \
  --reconstruction-samples 20                           \
  --latent-samples 256                                  \
  --generative-samples 10                               \
  --confusion-matrix                                    \
  --gpus 2 3                                            \
  --save-figures-to ~/figures                           \
  --metadata /data1/peter/metadata/BBBC021_v1_image.csv \
  --labels /data1/peter/metadata/BBBC021_v1_moa.csv     \
  --images /data1/peter/segmented                       \
  --cell-count-file /data1/peter/cell_counts.csv        \
  --restore-from /data1/peter/runs/checkpoints          \
