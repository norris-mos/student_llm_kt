import json
import torch
import pickle
from prompts import SPECIAL_TOKENS
from LoRa_preprocessing import StudentInteractionsDataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
from datasets import Dataset
import sys
import os

def finetune_model(config_path):
    """
    Train a language model using configuration specified in a JSON file.
    
    Args:
        config_path (str): Path to JSON configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Model initialization parameters
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config['model_name'],
        max_seq_length=config['max_seq_length'],
        dtype=config.get('dtype', None),  # None for auto detection
        load_in_4bit=config.get('load_in_4bit', True),
        token=config.get('hf_token', None)
    )

    # PEFT model configuration
    model = FastLanguageModel.get_peft_model(
        model,
        r=config['peft_config']['r'],
        target_modules=config['peft_config']['target_modules'],
        lora_alpha=config['peft_config']['lora_alpha'],
        lora_dropout=config['peft_config']['lora_dropout'],
        bias=config['peft_config']['bias'],
        use_gradient_checkpointing=config['peft_config']['use_gradient_checkpointing'],
        random_state=config['peft_config']['random_state'],
        use_rslora=config['peft_config']['use_rslora'],
        loftq_config=config['peft_config'].get('loftq_config', None)
    )

    print(model.print_trainable_parameters())
    print(tokenizer.eos_token)

    # Load training data

    try:

        with open(config['data_path'], 'rb') as f:
            loaded_dict = pickle.load(f)
    except:
        loaded_dict = torch.load(config['data_path']) 

        


    # Initialize dataset
    dataset = StudentInteractionsDataset(
        loaded_dict,
        tokenizer,
        config['max_seq_length'],
        cache_path=config['data_path']
    )
    train_data = dataset.load_data()
        # Convert to format expected by SFTTrainer
    # formatted_dataset = Dataset.from_dict({
    #     'text': train_data
    # })


    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=config['training_args']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training_args']['gradient_accumulation_steps'],
        warmup_steps=config['training_args']['warmup_steps'],
        max_steps=config['training_args']['max_steps'],
        learning_rate=config['training_args']['learning_rate'],
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=config['training_args']['logging_steps'],
        optim=config['training_args']['optim'],
        weight_decay=config['training_args']['weight_decay'],
        lr_scheduler_type=config['training_args']['lr_scheduler_type'],
        seed=config['training_args']['seed'],
        output_dir=config['training_args']['output_dir'],
        report_to=config['training_args']['report_to']
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        dataset_text_field="text",
        max_seq_length=config['max_seq_length'],
        dataset_num_proc=config['dataset_num_proc'],
        packing=config['packing'],
        args=training_args
    )

    # Start training
    trainer.train()

    model.save_pretrained('debug_model')
    tokenizer.save_pretrained('debug_tokenizer')

    return model, tokenizer