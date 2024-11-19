import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional, Union

class DynamicTransformerDKT(nn.Module):
    def __init__(self, 
                 feature_config: Dict[str, Dict],
                 hidden_dim: int,
                 time_dim: int = 2,
                 nhead: int = 2, 
                 num_layers: int = 1, 
                 dropout: float = 0.1):
        """
        Args:
            feature_config: Dictionary specifying features to include and their configs
            Example:
            {
                'question_id': {
                    'type': 'embedding',
                    'num_embeddings': 1000,
                    'embedding_dim': 64
                },
                'misconceptions': {
                    'type': 'embedding_with_mask',
                    'num_embeddings': 100,
                    'embedding_dim': 64
                },
                'question_embedding': {
                    'type': 'pretrained',
                    'input_dim': 768
                },
                'time_features': {
                    'type': 'continuous',
                    'input_dim': 2
                }
            }
        """
        super().__init__()
        self.feature_config = feature_config
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        
        # Create embeddings and projections for each feature
        self.feature_layers = nn.ModuleDict()
        self.input_size = 0
        
        for feature_name, config in feature_config.items():
            if config['type'] == 'embedding':
                self.feature_layers[feature_name] = nn.Embedding(
                    config['num_embeddings'],
                    config['embedding_dim']
                )
                self.input_size += config['embedding_dim']
                
            elif config['type'] == 'embedding_with_mask':
                self.feature_layers[f"{feature_name}_embed"] = nn.Embedding(
                    config['num_embeddings'],
                    config['embedding_dim']
                )
                self.input_size += config['embedding_dim']
                
            elif config['type'] == 'pretrained':
                if config.get('project', False):
                    self.feature_layers[f"{feature_name}_proj"] = nn.Linear(
                        config['input_dim'],
                        config['output_dim']
                    )
                    self.input_size += config['output_dim']
                else:
                    self.input_size += config['input_dim']
                    
            elif config['type'] == 'continuous':
                self.input_size += config['input_dim']

        # Model components
        self.input_proj = nn.Linear(self.input_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim*4,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Output layers
        output_input_size = hidden_dim
        if 'question_id' in feature_config:
            output_input_size += feature_config['question_id']['embedding_dim']
        if 'time_features' in feature_config:
            output_input_size += feature_config['time_features']['input_dim']
            
        self.output_layers = nn.Sequential(
            nn.Linear(output_input_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_config['correct_options']['num_embeddings'])
        )
        
        self.device = torch.device("cuda:0")
        self.to(self.device)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def generate_causal_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)

    def process_feature(self, feature_name: str, feature_data: torch.Tensor, 
                       mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process a single feature according to its configuration"""
        config = self.feature_config[feature_name]
        
        if config['type'] == 'embedding':
            return self.feature_layers[feature_name](feature_data)
            
        elif config['type'] == 'embedding_with_mask':
            embeddings = self.feature_layers[f"{feature_name}_embed"](feature_data)
            if mask is not None:
                embeddings = embeddings * mask.unsqueeze(-1)
            return embeddings.sum(dim=2)
            
        elif config['type'] == 'pretrained':
            if config.get('project', False):
                return self.feature_layers[f"{feature_name}_proj"](feature_data)
            return feature_data
            
        elif config['type'] == 'continuous':
            return feature_data
            
        raise ValueError(f"Unknown feature type: {config['type']}")

    def forward(self, batch: Dict[str, torch.Tensor]):
        # Move all inputs to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Process each feature
        processed_features = []
        for feature_name, config in self.feature_config.items():
            if feature_name == 'options':
                continue
                
            feature_data = batch[config.get('input_key', feature_name)]
            mask_key = config.get('mask_key')
            mask = batch.get(mask_key) if mask_key else None
            
            processed = self.process_feature(feature_name, feature_data, mask)
            processed_features.append(processed)

        # Prepare transformer input
        batch_size, seq_len = processed_features[0].shape[:2]
        transformer_input = [tensor.view(batch_size, seq_len, -1) 
                           for tensor in processed_features]
        transformer_input = torch.cat(transformer_input, dim=-1)
        
        # Project and transform
        transformer_input = self.input_proj(transformer_input)
        transformer_input = self.pos_encoder(transformer_input)
        
        # Apply transformer layers with causal mask
        causal_mask = self.generate_causal_mask(transformer_input.size(1))
        for layer in self.transformer_layers:
            transformer_input = layer(transformer_input, src_mask=causal_mask)
            
        # Shift outputs for prediction
        shifted_output = torch.cat([
            torch.zeros_like(transformer_input[:, :1, :]),
            transformer_input[:, :-1, :]
        ], dim=1)
        
        # Prepare prediction input
        pred_inputs = [shifted_output]
        if 'question_id' in self.feature_config:
            pred_inputs.append(self.process_feature('question_id', 
                             batch['question_ids']))
        if 'time_features' in self.feature_config:
            pred_inputs.append(batch['time_steps'])
            
        pred_input = torch.cat(pred_inputs, dim=-1)
        output = self.output_layers(pred_input)
        
        # Calculate loss
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        selected_options = batch['selected_options']
        mask = batch['mask']
        
        loss = loss_fn(output.view(-1, self.feature_config['correct_options']['num_embeddings']), 
                      selected_options.view(-1))
        masked_loss = loss.view(mask.shape) * mask
        loss = masked_loss.sum() / mask.sum()
        
        return loss, torch.softmax(output, dim=-1)

    def predict_option(self, batch):
        with torch.no_grad():
            loss, option_probs = self.forward(batch)
            predicted_options = torch.argmax(option_probs, dim=-1)
        return predicted_options.cpu().numpy()
    


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src
