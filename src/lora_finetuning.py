import sys
import json
import torch
import pickle
from prompts import SPECIAL_TOKENS
from LoRa_preprocessing import StudentInteractionsDataset
from transformers.trainer_callback import TrainerCallback,EarlyStoppingCallback
from sklearn.model_selection import KFold,train_test_split
import wandb
from datetime import datetime
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
from datasets import Dataset
import numpy as np
import os
# Define the custom WandbCallback class
class WandbCallback(TrainerCallback):
    def __init__(self, fold=None):
        self.fold = fold

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # If we have a fold number, add it to the metric names
            if self.fold is not None:
                logs = {f"fold_{self.fold+1}/{k}": v for k, v in logs.items()}
            wandb.log(logs)
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
        output_dir=config['output_dir'],
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

    model.save_pretrained('/mnt/ceph_rbd/data/models/qwen')
    tokenizer.save_pretrained('/mnt/ceph_rbd/data/models/qwen')

    return model, tokenizer





def finetune_model_cv(config_path, n_splits=5, project_name="model-finetuning"):
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

    
    # Initialize W&B run
    run_name = f"{config['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project=project_name,
        name=run_name,
        config={
            "n_splits": n_splits,
            **config
        }
    )
    
    best_metrics = {'eval_loss': float('inf')}
    best_model = None
    best_fold = None
    fold_results = []
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_data = np.array(train_data['text'])
    
    # Create a table for comparing fold results
    fold_comparison_table = wandb.Table(columns=["fold", "eval_loss", "train_loss", "epoch"])
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_data)):
            fold_output_dir = f"{config['output_dir']}/fold_{fold}"
            print(f"Training fold {fold+1}/{n_splits}")
            
            # Split the training data into training and in-training validation
            train_texts = [all_data[i] for i in train_idx]
            holdout_texts = [all_data[i] for i in val_idx]  # This is our cross-validation holdout set
            
            # Further split train_texts into train and in-training validation
            train_texts_final, in_training_val_texts = train_test_split(
                train_texts, 
                test_size=0.1,  # Use 10% of training data for in-training validation
                random_state=42 + fold  # Different seed for each fold
            )
            
            # Create our three datasets
            train_dataset = Dataset.from_dict({"text": train_texts_final})
            in_training_val_dataset = Dataset.from_dict({"text": in_training_val_texts})
            holdout_dataset = Dataset.from_dict({"text": holdout_texts})
       


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

            # Set up training arguments with proper validation strategy
            training_args = TrainingArguments(
                **config['training_args'],
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                output_dir=fold_output_dir,
                # Enable validation during training
                evaluation_strategy="steps",     # Validate every n steps
                eval_steps=100,                  # Adjust based on your dataset size
                save_strategy="steps",
                save_steps=100,
                load_best_model_at_end=True,     # Load best model based on in-training validation
                metric_for_best_model="loss",
                greater_is_better=False,
            )
            
            # Initialize trainer with in-training validation set
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=in_training_val_dataset,  # Use in-training validation set here
                dataset_text_field="text",
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=3,
                        early_stopping_threshold=0.01
                    ),
                    WandbCallback(fold=fold)
                ],
                args=training_args,
            )
            
            # Train the model (this will use in-training validation)
            train_result = trainer.train()
            train_metrics = train_result.metrics
            
            # Now evaluate on the hold-out set for this fold
            holdout_metrics = trainer.evaluate(
    eval_dataset=Dataset.from_dict({"text": holdout_texts}),
    metric_key_prefix="holdout"
)
            
            # Combine metrics
            fold_metrics = {
                'train_loss': train_metrics.get('train_loss', float('inf')),
                'in_training_val_loss': train_metrics.get('eval_loss', float('inf')),
                'holdout_loss': holdout_metrics.get('loss', float('inf')),
                'epoch': train_metrics.get('epoch', 0)
            }
            
            # Use holdout loss for selecting best model across folds
            if fold_metrics['holdout_loss'] < best_metrics['holdout_loss']:
                best_metrics = fold_metrics.copy()
                best_fold = fold
                
                # Save the best model
                best_model_dir = f"{config['output_dir']}/best_model"
                trainer.save_model(best_model_dir)
                
            print(f"\nFold {fold+1} Results:")
            print(f"Train Loss: {fold_metrics['train_loss']:.4f}")
            print(f"In-Training Validation Loss: {fold_metrics['in_training_val_loss']:.4f}")
            print(f"Hold-out Validation Loss: {fold_metrics['holdout_loss']:.4f}")