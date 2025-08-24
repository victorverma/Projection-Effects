#!/bin/sh

set -e
max_workers=${1:--1}
python test_model.py $max_workers
