#!/bin/bash
python test.py --name rfur --neighbour_slice 8 --checkpoints_dir checkpoints --model ft3_op.pth --case Case01002 --group_size 8  --gpus 0 \
    --root_dir /home/leofer/experiment/freehand/freehand
