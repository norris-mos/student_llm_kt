

conda create --name unsloth_env \
 python=3.10 \
 pytorch-cuda=12.1 \
 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
 -y
conda activate unsloth_env

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes

pip install wandb

wandb login 'e53412e1c10637ccea9fed89294a9a81a38c8579'

##### to get the model to finetune

apt-get update
apt-get install build-essential

export CC=gcc
export CXX=g++
