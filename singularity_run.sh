#!/usr/bin/env bash

if [ -z "$2" ]; then
    echo "Missing parameters."
    echo "Usage: $0 cuda_device config_file [main_file (default: train_eval.py) [output_filename]]"
    exit -1
fi

if [ -z "$3" ]; then
    main_file="train_eval.py"
else
    main_file="$3"
fi

container_file="dl-of-gm.sif"

if [ ! -f "$container_file" ]; then
    echo "Building singularity container to $container_file ..."
    singularity build --fakeroot "$container_file" singularity.def
fi

if [ -z "$4" ]; then
    singularity run \
      --nv \
      --bind /data:/data \
      "$container_file" \
      bash -c "CUDA_VISIBLE_DEVICES=$1 python $main_file --cfg $2"
else
    nohup \
      singularity run \
      --nv \
      --bind /data:/data \
      "$container_file" \
      bash -c "CUDA_VISIBLE_DEVICES=$1 python $main_file --cfg $2" \
      &>$4 \
      &
fi
