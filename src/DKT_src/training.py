import sys
sys.path.append('/mnt/ceph_rbd/LoRa/student_llm_kt/src')
from LoRa_preprocessing import DataFrame2InteractionDictionary, load_data
sys.path.append('/mnt/ceph_rbd/LoRa/student_llm_kt/src/DKT_src')
from models import DynamicTransformerDKT
from dataloader_new import SequenceDataset, options_dataloader_preproc_process
from tqdm import tqdm
import torch
from sklearn.metrics import precision_score, recall_score
from torch.utils.data import DataLoader
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
import os
import argparse
import json

from tqdm import tqdm

def train_dynamic_kt(train_data, val_data, feature_config, model_config, training_config, experiment_name="Dynamic_KT"):
   wandb.init(project="Dynamic_KT", name=experiment_name,
              config={"features": feature_config, "model": model_config, "training": training_config})

   model = DynamicTransformerDKT(
       feature_config=feature_config,
       hidden_dim=model_config['hidden_dim'],
       nhead=model_config.get('nhead', 2), 
       num_layers=model_config.get('num_layers', 1),
       dropout=model_config.get('dropout', 0.1)
   )

   train_loader = DataLoader(
       SequenceDataset(train_data),
       batch_size=training_config['batch_size'],
       shuffle=True,
       collate_fn=options_dataloader_preproc_process,
       num_workers=training_config.get('num_workers', 4)
   )

   val_loader = DataLoader(
       SequenceDataset(val_data),
       batch_size=training_config['batch_size'],
       shuffle=False, 
       collate_fn=options_dataloader_preproc_process,
       num_workers=training_config.get('num_workers', 4)
   )

   optimizer = torch.optim.Adam(model.parameters(), lr=training_config.get('learning_rate', 0.001))
   best_val_metrics = {'f1_micro': 0}
   patience_counter = 0
   best_model_state = None

   epochs_pbar = tqdm(range(training_config['max_epochs']), desc='Epochs')
   for epoch in epochs_pbar:
       model.train()
       train_losses = []
       
       train_pbar = tqdm(train_loader, desc=f'Training batch', leave=False)
       for batch in train_pbar:
           optimizer.zero_grad()
           loss, _ = model(batch)
           loss.backward()
           torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
           optimizer.step()
           train_losses.append(loss.item())
           train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

       model.eval()
       val_metrics = evaluate_model(model, val_loader)
       
       metrics = {
           'epoch': epoch,
           'train_loss': np.mean(train_losses),
           'val_loss': val_metrics['loss'],
           'val_accuracy': val_metrics['accuracy'],
           'val_f1_micro': val_metrics['f1_micro'],
           'val_f1_macro': val_metrics['f1_macro']
       }
       wandb.log(metrics)
       epochs_pbar.set_postfix(metrics)

       if val_metrics['f1_micro'] > best_val_metrics['f1_micro']:
           best_val_metrics = val_metrics
           best_model_state = model.state_dict()
           patience_counter = 0
       else:
           patience_counter += 1

       if patience_counter >= training_config['patience']:
           print(f"Early stopping triggered at epoch {epoch}")
           break

   if training_config.get('save_model', False):
       save_path = os.path.join(training_config['save_dir'], f"{feature_config.keys()}_best.pt")
       torch.save(best_model_state, save_path)
       wandb.save(save_path)

   wandb.log({
       'best_val_accuracy': best_val_metrics['accuracy'],
       'best_val_f1_micro': best_val_metrics['f1_micro'],
       'best_val_f1_macro': best_val_metrics['f1_macro']
   })
   
   wandb.finish()
   return best_model_state, best_val_metrics

def evaluate_model(model, dataloader):
   model.eval()
   all_labels, all_predictions, all_masks = [], [], []
   total_loss = 0
   
   val_pbar = tqdm(dataloader, desc='Validating', leave=False)
   with torch.no_grad():
       for batch in val_pbar:
           loss, logits = model(batch)
           predictions = torch.argmax(logits, dim=-1)
           
           all_labels.append(batch['selected_options'].cpu())
           all_predictions.append(predictions.cpu())
           all_masks.append(batch['mask'].cpu())
           total_loss += loss.item()
           val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

   labels = torch.cat([x.flatten() for x in all_labels]).numpy()
   preds = torch.cat([x.flatten() for x in all_predictions]).numpy()
   masks = torch.cat([x.flatten() for x in all_masks]).numpy()

   valid_indices = masks.astype(bool)
   valid_labels = labels[valid_indices]
   valid_preds = preds[valid_indices]

   return {
       'loss': total_loss / len(dataloader),
       'accuracy': accuracy_score(valid_labels, valid_preds),
       'f1_micro': f1_score(valid_labels, valid_preds, average='micro'),
       'f1_macro': f1_score(valid_labels, valid_preds, average='macro')
   }

def evaluate_model_class(model, dataloader):
   model.eval()
   all_labels, all_predictions, all_masks = [], [], []
   total_loss = 0
   
   val_pbar = tqdm(dataloader, desc='Validating', leave=False)
   with torch.no_grad():
       for batch in val_pbar:
           loss, logits = model(batch)
           predictions = torch.argmax(logits, dim=-1)
           
           all_labels.append(batch['selected_options'].cpu())
           all_predictions.append(predictions.cpu()) 
           all_masks.append(batch['mask'].cpu())
           total_loss += loss.item()
           val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

   labels = torch.cat([x.flatten() for x in all_labels]).numpy()
   preds = torch.cat([x.flatten() for x in all_predictions]).numpy()
   masks = torch.cat([x.flatten() for x in all_masks]).numpy()

   valid_indices = masks.astype(bool)
   valid_labels = labels[valid_indices] 
   valid_preds = preds[valid_indices]

   # Get per-class metrics
   precision = precision_score(valid_labels, valid_preds, average=None)
   recall = recall_score(valid_labels, valid_preds, average=None)
   f1 = f1_score(valid_labels, valid_preds, average=None)

   return {
       'loss': total_loss / len(dataloader),
       'accuracy': accuracy_score(valid_labels, valid_preds),
       'f1_micro': f1_score(valid_labels, valid_preds, average='micro'),
       'f1_macro': f1_score(valid_labels, valid_preds, average='macro'),
       'precision_per_class': precision,
       'recall_per_class': recall, 
       'f1_per_class': f1
   }

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
    answers,questions,misconceptions,question_subject = load_data('/mnt/ceph_rbd/LoRa/data')

    if config["training_config"]["train_cache"]:
        print("Loading train set from cache")

        all_training_data =  torch.load(config["training_config"]["train_cache"])
        train_data = {i: v for i, (k, v) in enumerate(list(all_training_data.items())[:-100])}
        val_data = {i: v for i, (k, v) in enumerate(list(all_training_data.items())[-100:])}
        
        

    else:
        #load data if not cached
        print("Creating train set from scratch")
        data = DataFrame2InteractionDictionary(answers,questions,misconceptions,question_subject,train_split=0.1)
        data.createedi(3456784,'/mnt/ceph_rbd/LoRa/filtered_interaction_dictionaries/eedi_train2.pt')




    best_model_state, best_metrics = train_dynamic_kt(
    train_data=train_data,
    val_data=val_data,
    feature_config=config["feature_config"],
    model_config=config["model_config"],
    training_config=config["training_config"],
    experiment_name="DKT_experiment"
)
    
     
    model = DynamicTransformerDKT(
        feature_config=config["feature_config"],
        hidden_dim=config["model_config"]['hidden_dim'],
        nhead=config["model_config"].get('nhead', 2),
        num_layers=config["model_config"].get('num_layers', 1),
        dropout=config["model_config"].get('dropout', 0.1)
    )
    model.load_state_dict(best_model_state)
    
    test_loader = DataLoader(
        SequenceDataset(test_data),
        batch_size=config["training_config"]['batch_size'],
        shuffle=False,
        collate_fn=options_dataloader_preproc_process,
        num_workers=config["training_config"].get('num_workers', 4)
    )
    
    test_metrics = evaluate_model_class(model, test_loader)
    print("\nTest Set Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":

    main()
# Example usage:
