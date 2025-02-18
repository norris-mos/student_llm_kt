from unsloth import FastLanguageModel
import json
import sys
import re
import os
from typing import Tuple, Optional
sys.path.append('/mnt/ceph_rbd/student_llm_kt/src')
sys.path.append('/mnt/ceph_rbd/student_llm_kt/src/DKT_src')
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

def generate_response_clean(model, tokenizer, input_text, max_new_tokens=1):
    """
    Generate multiple choice responses using unsloth FastLanguageModel
    Returns the last token generated with proper memory management
    """
    try:
        # Prepare inputs
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=40000
        ).to(model.device)
        
        # Generate
        with torch.no_grad():  # Prevent gradient computation
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
    
    finally:
        # Clean up tensors
        del inputs, outputs, last_tokens
        torch.cuda.empty_cache()

def generate_response_deterministic(model, tokenizer, batch_texts,task="options"):
    """
    Generate single token multiple choice responses (A,B,C,D only) for a batch of texts
    """
    try:
        # Prepare inputs
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=40000
        ).to(model.device)

        for i, text in enumerate(batch_texts):
            print(f"Sequence {i} tokens: {len(inputs['input_ids'][i])}")

        if task == "options":
            # Get the token IDs for A, B, C, D
            option_tokens = torch.tensor(tokenizer.convert_tokens_to_ids(['A', 'B', 'C', 'D'])).to(model.device)
        else:
            option_tokens = torch.tensor(tokenizer.convert_tokens_to_ids(['1','0'])).to(model.device)
        
        # Single forward pass for entire batch
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Get logits for the next token for all sequences
            next_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            
            # Only consider logits for A, B, C, D tokens
            option_logits = next_token_logits[:, option_tokens]  # [batch_size, 4]
            
            # Get the indices of highest scoring tokens (relative to our option_tokens)
            best_option_indices = torch.argmax(option_logits, dim=1)  # [batch_size]
            
            # Use advanced indexing to get the actual token ids
            selected_tokens = option_tokens[best_option_indices]  # [batch_size]
            
            # Add sequence dimension for tokenizer
            selected_tokens = selected_tokens.unsqueeze(-1)  # [batch_size, 1]
            
            # Decode all tokens in batch
            responses = tokenizer.batch_decode(selected_tokens, skip_special_tokens=True)
            
            
            return responses
            
    finally:
        # Clean up tensors
        del inputs, outputs, next_token_logits, option_logits, selected_tokens
        torch.cuda.empty_cache()



def process_dataset_in_batches_clean(model, tokenizer, dataset, batch_size=8,task="binary"):
    """
    Process entire dataset in batches with memory management
    """
    all_predictions = []
    
    valid_answers = {'A', 'B', 'C', 'D','1','0'}
    invalid_predictions = []
    
    def clean_prediction(pred):
        return pred.strip().upper() if isinstance(pred, str) else pred

    def standardize_prediction(pred):
        cleaned = clean_prediction(pred)
        print(cleaned)
        return cleaned if cleaned in valid_answers else None
    
    try:
        # Process data in batches
        for i in range(0, len(dataset), batch_size):
            # Clear cache at the start of each batch
            torch.cuda.empty_cache()
            
            # Get current batch
            batch = dataset[i:i + batch_size]
            
            # Generate predictions for batch
            batch_predictions = generate_response_deterministic(model, tokenizer, batch,task=task)
            batch_test = generate_response_clean(model,tokenizer,batch)
          

       
            
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
                    batch_predictions[idx] = 'NAN'
                else:
                    batch_predictions[idx] = cleaned_pred
            
            all_predictions.extend(batch_predictions)
            
            # Print progress and memory stats
            print(f"Processed {i + len(batch)}/{len(dataset)} samples")
            print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            
            # Optional: Force garbage collection
            if i % (batch_size * 10) == 0:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"Error during batch processing: {e}")
        raise
    
    finally:
        # Final cleanup
        torch.cuda.empty_cache()
    
    return all_predictions


def calculate_classification_metrics(y_true, y_pred, classification_type='options', class_labels=None):
    """
    Calculate classification metrics while properly handling NaN values.
    
    Parameters:
    -----------
    y_true : list or array-like
        Ground truth labels
    y_pred : list or array-like
        Predicted labels
    classification_type : str
        Type of classification: 'multiclass' or 'binary'
    class_labels : list
        List of class labels. If None:
        - For multiclass: defaults to ['A', 'B', 'C', 'D']
        - For binary: defaults to ['0', '1']
    
    Returns:
    --------
    dict
        Dictionary containing various classification metrics
    """
    # Set default class labels if not provided
    if class_labels is None:
        class_labels = ['A', 'B', 'C', 'D'] if classification_type == 'options' else ['0', '1']
    
    # Convert inputs to numpy arrays and ensure string type
    y_true = np.array([str(label) for label in y_true])
    y_pred = np.array([str(label) for label in y_pred])
    class_labels = [str(label) for label in class_labels]
    
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
            labels=class_labels,
            zero_division=0
        )
        
        # Store per-class metrics
        class_metrics = {}
        for idx, class_label in enumerate(class_labels):
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
            labels=class_labels
        )
        
    return metrics

def print_classification_report(metrics, classification_type='options'):
    """
    Print a formatted classification report.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing classification metrics
    classification_type : str
        Type of classification: 'multiclass' or 'binary'
    """
    print("Classification Report")
    print("=" * 50)
    print(f"Type: {classification_type.capitalize()}")
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Valid Predictions: {metrics['valid_predictions']}")
    print(f"NaN Predictions: {metrics['nan_predictions']} ({metrics['nan_rate']:.2%})")
    
    if metrics['valid_predictions'] > 0:
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
        print(f"{'':<10} {macro['precision']:>10.2%} {macro['recall']:>10.2%} {macro['f1_score']:>10.2%}")
        print("Weighted Average:")
        weighted = metrics['weighted_avg']
        print(f"{'':<10} {weighted['precision']:>10.2%} {weighted['recall']:>10.2%} {weighted['f1_score']:>10.2%}")
        
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
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
    data_path = os.path.join('/mnt/ceph_rbd/data')
    data_path = os.path.normpath(data_path)  # Normalize the path to remove any redundant parts
    questions = os.path.join(data_path,'questions.csv')
    answers = os.path.join(data_path,'answer.csv')
    question_subject = os.path.join(data_path,'question-subject.csv')
    misconception= os.path.join(data_path,'misconception.csv')
    questions = pd.read_csv(questions)
    answers = pd.read_csv(answers)
    question_subject = pd.read_csv(question_subject)
    misconception = pd.read_csv(misconception)
    return answers,questions,misconception,question_subject


def inference(model,tokenizer,config):




    
    answers, questions, misconception, question_subject = load_data()

    loaded_dict = DataFrame2InteractionDictionary(answers, questions, misconception, question_subject, train_split=0.9, random_seed=42)
    loaded_dict.createTestDict()

    FastLanguageModel.for_inference(model)

    dataset = StudentInteractionsDataset(
        loaded_dict.test_dictionary,
        tokenizer,
        config['max_seq_length'],
        cache_path=config["test_data"]
    )

    test_data = dataset.load_test_data()

    test_texts = test_data['text'][:5]
    ground_truth = test_data['response'][:5]  # Make sure this matches your dataset's true labels

    print(f"Doing inference over {len(test_texts)} examples")
    predictions = process_dataset_in_batches_clean(model, tokenizer, test_texts, batch_size=2)

    # Calculate metrics
    metrics = calculate_classification_metrics(ground_truth, predictions)

    # Print detailed report
    print_classification_report(metrics)

    # Save predictions and metrics
    results_dir = f'{config["output_dir"]}'
    os.makedirs(results_dir, exist_ok=True)

    # Save predictions
    predictions_df = pd.DataFrame({
        'ground_truth': ground_truth,
        'prediction': predictions
    })
    predictions_path = os.path.join(results_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)

    # Save metrics as JSON
    metrics_path = os.path.join(results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    # Save confusion matrix visualization
    if 'confusion_matrix' in metrics:
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            metrics['confusion_matrix'],
            annot=True,
            fmt='d',
            xticklabels=['A', 'B', 'C', 'D'],
            yticklabels=['A', 'B', 'C', 'D']
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        cm_path = os.path.join(results_dir, 'confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()

    print(f"\nResults saved to {results_dir}:")
    print(f"- Predictions: {predictions_path}")
    print(f"- Metrics: {metrics_path}")
    print(f"- Confusion Matrix: {cm_path}")


def main():




    args = parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    answers, questions, misconception, question_subject = load_data()

    loaded_dict = DataFrame2InteractionDictionary(answers, questions, misconception, question_subject, train_split=0.9, random_seed=42)
    if config["task"] == "binary":
        print(f'Carrying out binary KT task')
        loaded_dict.createTestDictBinary()

    else:
        loaded_dict.createTestDict()

    model, tokenizer = load_model_for_inference(
        model_path=config["inference_checkpoint"],
        tokenizer_path=config["inference_checkpoint"],
        config_path=args.config
    )
    FastLanguageModel.for_inference(model)

    dataset = StudentInteractionsDataset(
        loaded_dict.test_dictionary,
        tokenizer,
        config['max_seq_length'],
        cache_path=config["test_data"]
    )
    print(config['test_data'])

    test_data = dataset.load_test_data(task=config["task"])

    test_texts = test_data['text']
    
    ground_truth = test_data['response'] # Make sure this matches your dataset's true labels

    print(f"Doing inference over {len(test_texts)} examples")
    predictions = process_dataset_in_batches_clean(model, tokenizer, test_texts, batch_size=1,task=config["task"])
    print(predictions)
    print(ground_truth)

    # Calculate metrics
    metrics = calculate_classification_metrics(ground_truth, predictions,classification_type=config["task"])

    # Print detailed report
    
    print_classification_report(metrics)

    # Save predictions and metrics
    results_dir = config['output_dir']
    os.makedirs(results_dir, exist_ok=True)

    # Save predictions
    predictions_df = pd.DataFrame({
        'ground_truth': ground_truth,
        'prediction': predictions
    })
    predictions_path = os.path.join(results_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)

    # Save metrics as JSON
    metrics_path = os.path.join(results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    # Save confusion matrix visualization
    if 'confusion_matrix' in metrics:
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            metrics['confusion_matrix'],
            annot=True,
            fmt='d',
            xticklabels=['A', 'B', 'C', 'D'],
            yticklabels=['A', 'B', 'C', 'D']
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        cm_path = os.path.join(results_dir, 'confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()

    print(f"\nResults saved to {results_dir}:")
    print(f"- Predictions: {predictions_path}")
    print(f"- Metrics: {metrics_path}")
    print(f"- Confusion Matrix: {cm_path}")

if __name__ == "__main__":
    main()
