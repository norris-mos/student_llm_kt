from unsloth import FastLanguageModel
import json
import sys
import re
import os
from typing import Tuple, Optional
sys.path.append('../../../src')
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import pickle
import torch
from prompts import PROMPT_TEMPLATE_TEST, INSTRUCTION, SPECIAL_TOKENS
from LoRa_preprocessing import DataFrame2InteractionDictionary, StudentInteractionsDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
from unsloth import FastLanguageModel
import torch
from transformers import LogitsProcessor, LogitsProcessorList
import re
from torch.utils.data import DataLoader, Dataset
import argparse



def load_model_for_inference(
    model_path='debug_model_test',
    tokenizer_path='debug_tokenizer_test',
    config_path='path/to/your/config.json'  # same config used in training
):
    """
    Load the fine-tuned model and tokenizer for inference.
    
    Args:
        model_path (str): Path to saved model
        tokenizer_path (str): Path to saved tokenizer
        config_path (str): Path to configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        padding_side="right",
        truncation_side="right"
    )
    print('done loading tokenizer')
    
    # Add special tokens (same as in training)
    # tokenizer.add_special_tokens(SPECIAL_TOKENS)
    # print("Tokenizer vocabulary size:", len(tokenizer))

    # Load base model with same configuration as training
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config['model_name'],
        max_seq_length=config['max_seq_length'],
        dtype=config.get('dtype', None),
        load_in_4bit=config.get('load_in_4bit', True),
    )
    print('done loading model')
    
    # Resize embeddings
    # model.resize_token_embeddings(len(tokenizer))
    
    # Load the fine-tuned adapter weights
    model = PeftModel.from_pretrained(model, model_path)
    print('done loading peft')
  
    
    return model, tokenizer


def filter_long_context(tokenizer, data):
    token_lengths = [len(tokenizer.encode(text)) for text in data['text']]
    

    long_contexts = sum(1 for length in token_lengths if length > 40000)
    print(f"Number of texts exceeding 40,000 tokens: {long_contexts}")
    # Create list of (index, length) tuples
    indexed_lengths = list(enumerate(token_lengths))

    # Sort by length in descending order and get top 10
    top_10 = sorted(indexed_lengths, key=lambda x: x[1], reverse=True)[:10]

    # Print results
    print("Top 10 longest sequences:")
    for idx, length in top_10:
        print(f"Index {idx}: {length} tokens")
    
    # Optional: print all lengths to see distribution
    # for length in token_lengths:
    #     print(length)
    
    return long_contexts
 
    

  
  
    
    return model, tokenizer


def generate_response(model, tokenizer, input_text, max_new_tokens=1):
    """
    Generate multiple choice responses using unsloth FastLanguageModel
    Returns the last token generated
    """
    # Prepare inputs
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=40000
    ).to(model.device)
    
    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1
    )
    
    # Extract just the last token for each sequence
    last_tokens = outputs[:, -1:]  # Get the last token indices
    
    # Decode only the last tokens
    responses = tokenizer.batch_decode(last_tokens, skip_special_tokens=True)
    
    return responses


def process_dataset_in_batches(model, tokenizer, dataset, batch_size=8):
    """
    Process entire dataset in batches and collect predictions
    Validates that all predictions are one of {'A','B','C','D'}, ignoring whitespace and case
    """
    all_predictions = []
    valid_answers = {'A', 'B', 'C', 'D'}
    invalid_predictions = []
    
    def clean_prediction(pred):
        """Clean prediction by removing whitespace and converting to uppercase"""
        if isinstance(pred, str):
            return pred.strip().upper()
        return pred

    def standardize_prediction(pred):
        """Standardize prediction to exactly 'A', 'B', 'C', or 'D'"""
        cleaned = clean_prediction(pred)
        return cleaned if cleaned in valid_answers else None
    
    # Process data in batches
    for i in range(0, len(dataset), batch_size):
        # Get current batch
        batch = dataset[i:i + batch_size]
        
        # Generate predictions for batch
        batch_predictions = generate_response(model, tokenizer, batch)
        
        # Validate and clean predictions
        for idx, pred in enumerate(batch_predictions):
            cleaned_pred = standardize_prediction(pred)
            if cleaned_pred is None:
                print('None')
                invalid_predictions.append({
                    'batch_index': i + idx,
                    'original_prediction': pred,
                    'cleaned_prediction': clean_prediction(pred)
                })
                # Replace invalid prediction with a default value
                batch_predictions[idx] = 'NAN'
            else:
                batch_predictions[idx] = cleaned_pred
        
        all_predictions.extend(batch_predictions)
        
        # Optional: Print progress
        print(f"Processed {i + len(batch)}/{len(dataset)} samples")
    
    # Report invalid predictions if any were found
    if invalid_predictions:
        print("\nWarning: Found invalid predictions:")
        print(f"Total invalid predictions: {len(invalid_predictions)}")
        print("First few invalid predictions:")
        for inv in invalid_predictions[:5]:
            print(f"Index {inv['batch_index']}: '{inv['original_prediction']}' "
                  f"(cleaned: '{inv['cleaned_prediction']}')")
        print("All invalid predictions have been replaced with 'A'")
    
    return all_predictions


def calculate_classification_metrics(y_true, y_pred):
    """
    Calculate classification metrics while properly handling NaN values.
    
    Parameters:
    -----------
    y_true : list or array-like
        Ground truth labels (A, B, C, D)
    y_pred : list or array-like
        Predicted labels (A, B, C, D, NAN)
        
    Returns:
    --------
    dict
        Dictionary containing various classification metrics
    """
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Create a mask for valid predictions (non-NAN)
    valid_mask = y_pred != 'NAN'
    
    # Filter out NAN predictions and corresponding true values
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    # Calculate metrics
    metrics = {}
    
    # Basic counts
    metrics['total_samples'] = len(y_true)
    metrics['valid_predictions'] = np.sum(valid_mask)
    metrics['nan_predictions'] = len(y_true) - np.sum(valid_mask)
    metrics['nan_rate'] = metrics['nan_predictions'] / metrics['total_samples']
    
    # Calculate metrics for valid predictions
    if metrics['valid_predictions'] > 0:
        # Accuracy
        metrics['accuracy'] = accuracy_score(y_true_valid, y_pred_valid)
        
        # Precision, recall, and F1 score for each class
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_valid, 
            y_pred_valid,
            labels=['A', 'B', 'C', 'D'],
            zero_division=0
        )
        
        # Store per-class metrics
        class_metrics = {}
        for idx, class_label in enumerate(['A', 'B', 'C', 'D']):
            class_metrics[class_label] = {
                'precision': precision[idx],
                'recall': recall[idx],
                'f1_score': f1[idx],
                'support': support[idx]
            }
        metrics['class_metrics'] = class_metrics
        
        # Calculate macro and weighted averages
        metrics['macro_avg'] = {
            'precision': np.mean(precision),
            'recall': np.mean(recall),
            'f1_score': np.mean(f1)
        }
        
        metrics['weighted_avg'] = {
            'precision': np.average(precision, weights=support),
            'recall': np.average(recall, weights=support),
            'f1_score': np.average(f1, weights=support)
        }
        
        # Confusion Matrix
        metrics['confusion_matrix'] = confusion_matrix(
            y_true_valid, 
            y_pred_valid,
            labels=['A', 'B', 'C', 'D']
        )
        
    return metrics

def print_classification_report(metrics):
    """
    Print a formatted classification report.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing classification metrics
    """
    print("Classification Report")
    print("=" * 50)
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Valid Predictions: {metrics['valid_predictions']}")
    print(f"NaN Predictions: {metrics['nan_predictions']} ({metrics['nan_rate']:.2%})")
    print("\nOverall Accuracy: {:.2%}".format(metrics['accuracy']))
    print("\nPer-Class Metrics:")
    print("-" * 50)
    print(f"{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 50)
    
    for class_label, class_metric in metrics['class_metrics'].items():
        print(f"{class_label:<10} {class_metric['precision']:>10.2%} {class_metric['recall']:>10.2%} "
              f"{class_metric['f1_score']:>10.2%} {class_metric['support']:>10}")
    
    print("\nAverage Metrics:")
    print("-" * 50)
    print("Macro Average:")
    macro = metrics['macro_avg']
    print(f"{'':10} {macro['precision']:>10.2%} {macro['recall']:>10.2%} {macro['f1_score']:>10.2%}")
    print("Weighted Average:")
    weighted = metrics['weighted_avg']
    print(f"{'':10} {weighted['precision']:>10.2%} {weighted['recall']:>10.2%} {weighted['f1_score']:>10.2%}")

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

def load_data():
    data_path = os.path.join('/mnt/ceph_rbd/Process-Knowledge-Tracing/data')
    data_path = os.path.normpath(data_path)  # Normalize the path to remove any redundant parts
    questions = os.path.join(data_path,'questions.csv')
    answers = os.path.join(data_path,'answer.csv')
    question_subject = os.path.join(data_path,'question_subject.csv')
    misconception= os.path.join(data_path,'misconception.csv')
    questions = pd.read_csv(questions)
    answers = pd.read_csv(answers)
    question_subject = pd.read_csv(question_subject)
    misconception = pd.read_csv(misconception)
    return answers,questions,misconception,question_subject

def main():

    args = parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    answers,questions,misconception,question_subject = load_data()

    loaded_dict = DataFrame2InteractionDictionary(answers, questions, misconception, question_subject, train_split=0.9, random_seed=42)
    loaded_dict.createTestDict()

#     model,tokenizer = FastLanguageModel.from_pretrained(
#     model_name = config["inference_checkpoint"], # YOUR MODEL YOU USED FOR TRAINING
#     max_seq_length = config["max_seq_length"],
#     dtype = config.get('dtype', None),
#     load_in_4bit =True,
# )
    

    model, tokenizer = load_model_for_inference(
    model_path='/mnt/ceph_rbd/Process-Knowledge-Tracing/scripts/LoRa/max_interactions/inference_check/checkpoint-13000',
    tokenizer_path='/mnt/ceph_rbd/Process-Knowledge-Tracing/scripts/LoRa/max_interactions/debug_tokenizer',
    config_path='/mnt/ceph_rbd/Process-Knowledge-Tracing/scripts/LoRa/max_interactions/llama3.2-20000.json'
)
    FastLanguageModel.for_inference(model)

    dataset = StudentInteractionsDataset(
    loaded_dict.test_dictionary,
    tokenizer,
    config['max_seq_length'],
    cache_path=config["data_path"]
    
)
    test_data = dataset.load_test_data()
    tst = test_data['text']
    print(f"Doing inference over {len(tst)} examples")
    res = process_dataset_in_batches(model,tokenizer,tst,batch_size=20)


if __name__ == "__main__":
    main()
