#!/usr/bin/env bash

if [ -z "$3" ]; then
    echo "Missing parameters."
    echo "Usage: $0 container_name cuda_device config_file [main_file (default: train_eval.py)]"
    exit -1
fi

if [ -z "$4" ]; then
    main_file="train_eval.py"
else
    main_file="$4"
fi

docker run \
    --gpus "device=$2" \
    -d \
    -it \
    --ipc=host \
    --mount type=bind,source=/data/wangrunzhong/dl-of-gm,target=/workspace/dl-of-gm \
    --mount type=bind,source=/data/wangrunzhong/tmp/out,target=/workspace/dl-of-gm/output \
    --mount type=bind,source=/home/wangrunzhong/.cache,target=/.cache \
    -v /etc/localtime:/etc/localtime:ro \
    --user $(id -u):$(id -g) \
    -w /workspace/dl-of-gm \
    --name $1 \
    registry.cn-shanghai.aliyuncs.com/wangrunzhong/dl-of-gm:pytorch1.3_cu10.1 \
    /opt/conda/bin/python $main_file --cfg $3
