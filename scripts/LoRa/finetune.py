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
sys.path.append('/mnt/ceph_rbd/LoRa/student_llm_kt/src')
from LoRa_preprocessing import StudentInteractionsDataset
from lora_finetuning import finetune_model
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

  
    model,tokenizer = finetune_model(args.config)


if __name__ == "__main__":
    main()
