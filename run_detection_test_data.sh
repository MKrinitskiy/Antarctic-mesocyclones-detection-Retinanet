#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python detect.py --backbone=resnet152 --model-snapshot=./snapshots/resnet152_sail_20.h5 --interpolation-constants-directory=../train_data_collect/.cache/ --base-data-directory=/app/geoseg_server/src_data/nc_2012/ --output-directory=./output/ --batch-size=32 --proba-threshold=0.97