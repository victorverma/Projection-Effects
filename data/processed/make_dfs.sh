#!/bin/sh

for i in 1 2 3 4 5; do
    conda run -p ../../env/ --live-stream sh -c "
        echo 'Partition $i'
        python make_df.py --partition $i
        python make_df.py --partition $i --df_type 'summary'
    "
done
