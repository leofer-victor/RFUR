#!/bin/bash
python train.py --name rfur --neighbour_slice 8 --learning_rate 0.001 --batch_size 8 --epochs 500 --gpus 0   \
    --root_dir /home/leofer/experiment/freehand/freehand 