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
# docker://registry.cn-shanghai.aliyuncs.com/wangrunzhong/dl-of-gm:pytorch1.3_cu10.1 \

container_file="dl-of-gm.sif"

if [ ! -f "$container_file" ]; then
    echo "Building singularity container to $container_file ..."
    singularity build --fakeroot "$container_file" singularity.def
fi

nohup \
  singularity run \
  --nv \
  --bind /data:/data \
  "$container_file" \
  bash -c "CUDA_VISIBLE_DEVICES=$2 python $main_file --cfg $3" \
  &>$1 \
  &