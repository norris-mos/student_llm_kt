#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to print status messages
print_status() {
    echo "===> $1"
}

# Function to handle errors
handle_error() {
    print_status "Error occurred at line $1"
    exit 1
}

# Set error handler
trap 'handle_error $LINENO' ERR

# Set environment variables
export CC=gcc
export CXX=g++

# 1. Update package list and install system dependencies
print_status "Updating package list and installing dependencies"
apt-get update
apt-get install build-essential


# 2. Git operations
print_status "Performing Git operations"



# 3. PyTorch installation
print_status "Installing PyTorch"
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Conda installations
print_status "Installing Hugging Face Hub"
conda install -c conda-forge huggingface_hub -y

# 5. Creating and setting up Unsloth environment
print_status "Setting up Unsloth environment"
conda create --name unsloth_env \
    python=3.10 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers \
    -c pytorch -c nvidia -c xformers \
    -y

# 6. Activate Unsloth environment and install dependencies
print_status "Activating Unsloth environment and installing dependencies"
eval "$(conda shell.bash hook)"
conda activate unsloth_env
conda install gcc

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes

# 7. Login to wandb
print_status "Logging into Weights & Biases"
pip install wandb
wandb login 'e53412e1c10637ccea9fed89294a9a81a38c8579'

# 8. Run the script if ARG is provided
if [ ! -z "$ARG" ]; then
    print_status "Running script: $ARG"
    python "$ARG"
else
    print_status "No script specified (ARG not set)"
fi

print_status "Setup completed successfully"