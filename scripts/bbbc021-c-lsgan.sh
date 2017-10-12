#!/bin/bash

python3 -m cytogan.train.bbbc021                          \
  --epochs 50                                             \
  --model c-lsgan                                         \
  --lr 1e-5 7e-5                                          \
  --lr-decay 0.9995                                       \
  --lr-decay-steps 100                                    \
  --batch-size 128                                        \
  --workspace /data1/peter/runs                           \
  --checkpoint-freq '10min'                               \
  --summary-freq '10min'                                  \
  --generative-samples 100                                \
  --gpus 0 1                                              \
  --metadata /data1/peter/metadata/BBBC021_v1_image.csv   \
  --labels /data1/peter/metadata/BBBC021_v1_moa.csv       \
  --images /data1/peter/segmented                         \
  --cell-count-file /data1/peter/metadata/cell_counts.csv \
  --latent-compounds                                      \
  --latent-moa                                            \
  --confusion-matrix                                      \
  --normalize-luminance                                   \
  --save-profiles                                         \
  --vector-distance                                       \
  $@
