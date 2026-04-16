#!/bin/bash
sudo docker run --detach --tty \
  --gpus all \
  --name nemo_dev_$(date +%m%d%H%M) \
  --net host \
  --pid host \
  --privileged \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  --shm-size 80g \
  -v /home:/home \
  -v /data:/data \
  -v /scratch:/scratch \
  -v /mnt/vast:/mnt/vast \
  -e HF_HUB_CACHE=/data/johnson/huggingface/ \
  --entrypoint '/bin/bash' \
  nvcr.io/nvidia/nemo:26.02.00
