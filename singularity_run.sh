#!/usr/bin/env bash

if [ -z "$3" ]; then
    echo "Missing parameters."
    echo "Usage: $0 output_filename cuda_device config_file [main_file (default: train_eval.py)]"
    exit -1
fi

if [ -z "$4" ]; then
    main_file="train_eval.py"
else
    main_file="$4"
fi

#   ~/dl-of-gm_pytorch1.3_cu10.1.sif \

nohup \
  singularity run \
  --nv \
  --bind /data:/data \
  docker://registry.cn-shanghai.aliyuncs.com/wangrunzhong/dl-of-gm:pytorch1.3_cu10.1 \
  bash -c "CUDA_VISIBLE_DEVICES=$2 python $main_file --cfg $3" \
  &>$1 \
  &