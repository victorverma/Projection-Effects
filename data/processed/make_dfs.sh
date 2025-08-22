#!/bin/sh

set -e

for i in 1 2 3 4 5; do
    echo "Making data frames for partition $i..."
    conda run -p ../../env/ --live-stream sh -c "
        set -e
        python make_full_df.py --partition $i
        python correct_full_df.py --partition $i
        python summarize_full_df.py --partition $i
        python summarize_full_df.py --partition $i --use_corrected_data
    "
done
