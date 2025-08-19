#!/bin/sh

n_jobs=${1:--1}
for i in 1 2 3 4 5; do
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Training partition $i model..."
    conda run -p ../../env/ --live-stream python train_model.py \
        --partition $i \
        --n_jobs $n_jobs
done
