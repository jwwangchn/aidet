#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
GPUS=$2
PORT=${PORT:-53231}

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_addr BJ-IDC1-10-10-30-104 --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
