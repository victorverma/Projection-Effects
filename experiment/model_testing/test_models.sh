#!/bin/sh

set -e

for i in 1 2 3 4 5; do
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Testing partition $i models..."
    conda run -p ../../env/ --live-stream sh -c "
        set -e
        python test_model.py --train_partition $i
        python test_model.py --train_partition $i --use_corrected_data
    "
done
