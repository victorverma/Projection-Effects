#!/bin/sh

set -e
max_workers=${1:--1}
conda run -p ../../env/ --live-stream python test_models.py \
    --max_workers $max_workers
