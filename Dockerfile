FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-devel

MAINTAINER runzhong.wang

RUN apt-get update && apt-get install ninja-build
RUN pip install tensorboardX scipy easydict pyyaml
