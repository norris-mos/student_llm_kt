import sys
import json
import torch
import argparse
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from preproc_moocradar import StudentDataProcessor
from typing import Dict, List

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on MOOC data')
    parser.add_argument('config', type=str, help='Path to config file')
    parser.add_argument('interactions_path', type=str, help='Path to interactions JSONL file')
    parser.add_argument('questions_path', type=str, help='Path to questions JSONL file')
    return parser.parse_args()

def load_model_for_inference(
    model_path: str,
    tokenizer_path: str,
    config_path: str
):
    """
    Load the fine-tuned model and tokenizer for inference
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        padding_side="right",
        truncation_side="right"
    )
    print('Tokenizer loaded')

    # Load model
    model = FastLanguageModel.from_pretrained(
        model_path,
        max_seq_length=config['max_seq_length'],
        load_in_4bit=config['load_in_4bit'],
        token=config.get('hf_token', None)
    )
    
    FastLanguageModel.for_inference(model)
    print('Model loaded and prepared for inference')
    
    return model, tokenizer

def process_dataset_in_batches(model, tokenizer, dataset: Dict[str, List[str]], batch_size=8):
    """
    Process dataset in batches with memory management
    """
    all_predictions = []
    valid_answers = {'0', '1'}  # Binary classification
    
    def clean_prediction(pred: str) -> str:
        return pred.strip()

    try:
        texts = dataset['text']
        for i in range(0, len(texts), batch_size):
            torch.cuda.empty_cache()
            
            batch = texts[i:i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions = [clean_prediction(pred) for pred in decoded]
            
            # Validate predictions
            valid_preds = [pred if pred in valid_answers else None for pred in predictions]
            all_predictions.extend(valid_preds)
            
            if i % (batch_size * 10) == 0:
                print(f"Processed {i}/{len(texts)} examples")
                
    except Exception as e:
        print(f"Error during batch processing: {str(e)}")
        
    return all_predictions

def calculate_metrics(y_true: List[str], y_pred: List[str]):
    """
    Calculate classification metrics
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    # Filter out None predictions
    valid_indices = [i for i, pred in enumerate(y_pred) if pred is not None]
    y_true_filtered = [y_true[i] for i in valid_indices]
    y_pred_filtered = [y_pred[i] for i in valid_indices]
    
    accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_filtered, 
        y_pred_filtered, 
        average='binary'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'valid_predictions': len(y_pred_filtered),
        'total_examples': len(y_pred)
    }

def print_metrics(metrics: Dict):
    """
    Print formatted metrics
    """
    print("\nClassification Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Valid Predictions: {metrics['valid_predictions']}/{metrics['total_examples']}")

def main():
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Initialize data processor
    processor = StudentDataProcessor()
    processor.load_jsonl_data(args.interactions_path, args.questions_path)
    processor.create_test_sequences()
    
    # Load model and tokenizer
    model, tokenizer = load_model_for_inference(
        model_path=config["inference_checkpoint"],
        tokenizer_path=config["inference_checkpoint"],
        config_path=args.config
    )
    
    # Format test data
    test_data = processor.get_formatted_dataset()
    
    # Run inference
    print("\nRunning inference...")
    predictions = process_dataset_in_batches(
        model, 
        tokenizer, 
        test_data, 
        batch_size=config.get('inference_batch_size', 8)
    )
    
    # Calculate and print metrics
    metrics = calculate_metrics(test_data['response'], predictions)
    print_metrics(metrics)

if __name__ == "__main__":
    main()