from unsloth import FastLanguageModel
import torch
import pickle
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
from datasets import Dataset
from unsloth import is_bfloat16_supported
import sys
import os
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Process-Knowledge-Tracing
data_dir = os.path.dirname(project_root)  # Go up one more level to get to data

sys.path.append(os.path.join(data_dir,'src'))
sys.path.append(os.path.join(data_dir,'src/DKT_src'))

from LoRa_preprocessing import StudentInteractionsDataset
from lora_finetuning import finetune_model, finetune_model_cv
from finetuned_inference import inference
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Model evaluation script')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the configuration JSON file'
    )

    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)

  
    model,tokenizer = finetune_model_cv(args.config)

    # then inference
    inference(model,tokenizer,config)


if __name__ == "__main__":
    main()

