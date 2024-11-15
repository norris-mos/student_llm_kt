#!/bin/bash
##!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

ARG=$1
ARG2=$2

# 1. Update package list

apt-get update

# 2. Install git-lfs

apt-get install -y git-lfs

# 3. Pull the latest changes from git

git pull

# 4. Pull LFS files

git lfs pull

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

conda install -c conda-forge huggingface_hub
pip install unsloth

# 5. Log in to wandb (you'll need to handle authentication, possibly using secrets)

wandb login 'e53412e1c10637ccea9fed89294a9a81a38c8579'

cd /mnt/ceph_rbd/Process-Knowledge-Tracing/scripts/$ARG2

# 7. Run the script

python $ARG

########## for unsloth
conda create --name unsloth_env \
 python=3.10 \
 pytorch-cuda=12.1 \
 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
 -y
conda activate unsloth_env

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes

##### to get the model to finetune

apt-get update
apt-get install build-essential

export CC=gcc
export CXX=g++
