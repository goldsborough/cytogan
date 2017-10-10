#!/bin/bash

convert -background white -alpha remove -layers OptimizePlus -delay 20 \
  $1/*.png -loop 0 ${2:-animated.gif}
