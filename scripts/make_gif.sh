#!/bin/bash

convert -background white -alpha remove -layers OptimizePlus -delay 10 \
  "$1/%d.png[0-$(($(ls $1 | wc -l)-1))]" -loop 0 ${2:-animated.gif}
