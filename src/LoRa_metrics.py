import re
from finetuned_inference import generate_response
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset


VALID_ANSWERS = {'A', 'B', 'C', 'D'}

def extract_answer_before_token(text):


    cleaned_text = re.sub(r'[\n\s]+',' ',text)
    pattern = r'([ABCD])\s*<\|end_of_text\|>'
    match = re.search(pattern,cleaned_text)

    if match:
        answer = match.group(1)
        if answer in VALID_ANSWERS:
            return answer
    return None


def process_llm_response(response):
    """
    Processes an LLM response and extracts a valid answer (A, B, C, or D).
    
    Args:
        response: The full LLM response text
        
    Returns:
        A tuple containing:
            - The extracted answer (A, B, C, or D) or None
            - A success status boolean
    """
    try:
        answer = extract_answer_before_token(response)
        if answer in VALID_ANSWERS:
            print(answer)
            return answer, True
        
        return 'NA', True
    except Exception as e:
        print(f"Error processing response: {e}")
        return None, False



def calculate_accuracy_metrics(texts, response, model, tokenizer):
    preds = []
    
    # Wrap the loop with tqdm
    for text in tqdm(texts, desc="Processing texts", unit="text"):
        try:
            model_response = generate_response(model, tokenizer, text)
            prediction = process_llm_response(model_response)
            print(prediction[0])
            preds.append(prediction[0])
        except Exception as e:
            print(f"Error processing text: {str(e)}")
            preds.append(None)  # or handle the error as appropriate
    
    print(f"Predictions: {len(preds)}")
    print(f"Responses: {len(response)}")
    
    # Filter out None values if any errors occurred
    preds = [p for p in preds if p is not None]
    
    acc = evaluate_predictions(preds, response)
    return acc


class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]

def calculate_accuracy_metrics_batch(texts, response, model, tokenizer, batch_size=16):
    dataset = TextDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    
    for batch in tqdm(dataloader, desc="Processing batches", unit="batch"):
        print(batch)
        try:
            # Generate responses for the entire batch
            batch_responses = generate_batch_response(model, tokenizer, batch)
            print(f'print resopnses {batch_responses}')
            
            # Process each response in the batch
            batch_predictions = [process_llm_response(resp) for resp in batch_responses]
            print(batch_predictions)
            
            preds.extend([pred[0] for pred in batch_predictions])
            
            
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            # Add None values for the entire batch
            preds.extend([None] * len(batch))
    
    print(f"Predictions: {len(preds)}")
    print(f"Responses: {len(response)}")
    
    # Filter out None values if any errors occurred
    preds = [p for p in preds if p is not None]
    
    acc = evaluate_predictions(preds, response)
    return acc

def generate_batch_response(model, tokenizer, batch):
    # Assuming model supports batch processing
    # Modify this based on your model's specifics
    inputs = tokenizer(list(batch), padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(**inputs)
    
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)




    
def accuracy(preds, response):
    """
    Calculate accuracy score given model predictions and ground truth responses.
    
    Args:
        preds (list): List of model predictions (A, B, C, or D)
        response (list): List of ground truth responses (A, B, C, or D)
    
    Returns:
        float: Accuracy score between 0 and 1
        
    Example:
        >>> preds = ['A', 'B', 'C', 'A']
        >>> truth = ['A', 'B', 'D', 'A']
        >>> accuracy(preds, truth)
        0.75
    """
    if len(preds) != len(response):
        raise ValueError("Predictions and ground truth must have the same length")
    
    if not preds or not response:
        raise ValueError("Input lists cannot be empty")
    
    total = len(response)
    correct = sum(1 for p, r in zip(preds, response) if p == r)
    
    return correct / total

def evaluate_predictions(preds, response):
    """
    Extended evaluation function that provides more detailed metrics.
    
    Args:
        preds (list): List of model predictions
        response (list): List of ground truth responses
    
    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    if len(preds) != len(response):
        raise ValueError("Predictions and ground truth must have the same length")
        
    total = len(response)
    correct = 0
    per_class_correct = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    per_class_total = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    
    for p, r in zip(preds, response):
        if p == r:
            correct += 1
            per_class_correct[r] += 1
        per_class_total[r] += 1
    
    # Calculate per-class accuracies
    per_class_accuracy = {}
    for class_label in 'ABCD':
        if per_class_total[class_label] > 0:
            per_class_accuracy[class_label] = per_class_correct[class_label] / per_class_total[class_label]
        else:
            per_class_accuracy[class_label] = 0.0
    
    return {
        'overall_accuracy': correct / total,
        'per_class_accuracy': per_class_accuracy,
        'total_samples': total,
        'correct_predictions': correct
    }




