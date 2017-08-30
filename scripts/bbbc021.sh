#!/bin/bash

python3 -m cytogan.train.bbbc021                        \
  --epochs 10                                           \
  --model vae                                           \
  --lr 0.001                                            \
  --lr-decay 0.996                                      \
  --lr-decay-steps 100                                  \
  --batch-size 128                                      \
  --workspace /data1/peter/runs                         \
  --checkpoint-freq '1min'                              \
  --summary-freq '1min'                                 \
  --reconstruction-samples 20                           \
  --latent-samples 128                                  \
  --generative-samples 10                               \
  --gpus 2 3                                            \
# Specific to BBBC021
  --metadata /data1/peter/metadata/BBBC021_v1_image.csv \
  --labels /data1/peter/metadata/BBBC021_v1_moa.csv     \
  --images /data1/peter/segmented                       \
  --cell-count-file /data1/peter/cell_counts.csv        \
  --confusion-matrix                                    \
  $1
