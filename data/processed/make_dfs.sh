#!/bin/sh

set -e

for i in 1 2 3 4 5; do
    conda run -p ../../env/ --live-stream sh -c "
        set -e
        echo 'Partition $i'
        python make_full_df.py --partition $i
        python make_full_df.py --partition $i --correct_params
        python summarize_full_df.py --partition $i
        python summarize_full_df.py --partition $i --use_corrected_data
    "
done
