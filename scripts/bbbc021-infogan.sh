#!/bin/bash

python3 -m cytogan.train.bbbc021                          \
  --epochs 50                                             \
  --model infogan                                         \
  --lr 5e-4 2e-4 2e-4                                     \
  --lr-decay 0.999                                        \
  --lr-decay-steps 100                                    \
  --batch-size 64                                         \
  --workspace /data1/peter/runs                           \
  --checkpoint-freq '30s'                                 \
  --summary-freq '30s'                                    \
  --latent-samples 100                                    \
  --generative-samples 100                                \
  --gpus 0 1 2 3                                          \
  --metadata /data1/peter/metadata/BBBC021_v1_image.csv   \
  --labels /data1/peter/metadata/BBBC021_v1_moa.csv       \
  --images /data1/peter/segmented                         \
  --cell-count-file /data1/peter/metadata/cell_counts.csv \
  --latent-compounds                                      \
  --latent-moa                                            \
  --confusion-matrix                                      \
  $@
