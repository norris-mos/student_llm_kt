import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from typing import List, Dict, Union, NamedTuple
from datetime import datetime
from data_classes import EediItem
import ast
import math



class SequenceDataset(Dataset):
    def __init__(self, data):
        self.data = data
        print(len(data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):


        return self.data[idx]


class SequenceItem(NamedTuple):
    """Represents a single item in a sequence with its tensors"""
    question_id: torch.Tensor
    correctness: torch.Tensor
    time_features: torch.Tensor
    answer_value: torch.Tensor
    correct_answer: torch.Tensor
    question_embedding: torch.Tensor
    construct_id: torch.Tensor
    construct_mask: torch.Tensor

class SequencePreprocessor:
    def __init__(self):
        pass
    
    @staticmethod
    def encode_time_of_day(timestamp: float) -> float:
        """Encode time of day as cosine value"""
        seconds_in_day = 24 * 60 * 60
        time_of_day = timestamp % seconds_in_day
        return math.cos(time_of_day / seconds_in_day * 2 * math.pi)
    
    @staticmethod
    def encode_day_of_week(timestamp: float) -> float:
        """Encode day of week as cosine value"""
        seconds_in_week = 7 * 24 * 60 * 60
        start_of_week = timestamp - (datetime.fromtimestamp(timestamp).weekday() * 24 * 60 * 60)
        day_of_week = timestamp % seconds_in_week
        return math.cos(day_of_week / seconds_in_week * 2 * math.pi)
    
    def create_cosine_features(self, date_string: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Create cosine features from datetime string"""
        # Parse the datetime string
        try:
            # Try the format with milliseconds
            date_obj = datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            try:
                # Try without milliseconds
                date_obj = datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
            except ValueError as e:
                raise ValueError(f"Unable to parse date string: {date_string}") from e
        
        timestamp = date_obj.timestamp()
        cosine_time_of_day = torch.tensor(
            self.encode_time_of_day(timestamp), 
            dtype=torch.float
        ).reshape((1, 1))
        cosine_day_of_week = torch.tensor(
            self.encode_day_of_week(timestamp), 
            dtype=torch.float
        ).reshape((1, 1))
        return cosine_time_of_day, cosine_day_of_week

    def process_single_item(self, item: EediItem, max_construct: int) -> SequenceItem:
        """Process a single item in the sequence"""
        # Basic features
        correctness = torch.tensor(float(item.is_correct), dtype=torch.float)
        question_id = torch.tensor(item.question_id, dtype=torch.long)
        answer_value = torch.tensor(item.answer_value - 1, dtype=torch.long)
        correct_answer = torch.tensor(item.correct_answer - 1, dtype=torch.long)
        
        # Time features using the datetime field directly
        cosine_time, cosine_day = self.create_cosine_features(item.date_answered)
        time_features = torch.cat([cosine_time, cosine_day], dim=1)
        
        # Question embedding
        q_embed = torch.tensor(ast.literal_eval(item.question_embedding) if isinstance(item.question_embedding, str) else item.question_embedding)
        
        # Construct ID with padding
        construct_id = torch.tensor([item.construct_id], dtype=torch.long)
        construct_mask = torch.ones(1, dtype=torch.bool)
        construct_id = F.pad(construct_id, (0, max_construct - len(construct_id)), value=0)
        construct_mask = F.pad(construct_mask, (0, max_construct - len(construct_mask)), value=False)
        
        return SequenceItem(
            question_id=question_id,
            correctness=correctness,
            time_features=time_features,
            answer_value=answer_value,
            correct_answer=correct_answer,
            question_embedding=q_embed,
            construct_id=construct_id,
            construct_mask=construct_mask
        )

    def process_sequence(self, sequence: List[EediItem]) -> Dict[str, torch.Tensor]:
        """Process a single sequence of items"""
        # Sort sequence by timestamp
        sorted_items = sorted(sequence, key=lambda x: x.date_answered)
        
        # Get max length for padding construct IDs
        max_construct = 1
        
        # Process each item
        processed_items = [
            self.process_single_item(item, max_construct)
            for item in sorted_items
        ]
        
        # Stack tensors
        return {
            'questionids': torch.stack([item.question_id for item in processed_items]),
            'labels': torch.stack([item.correctness for item in processed_items]),
            'time_steps': torch.stack([item.time_features for item in processed_items]),
            'QuestionEmbedding': torch.stack([item.question_embedding for item in processed_items]),
            'construct_ids': torch.stack([item.construct_id for item in processed_items]),
            'construct_mask': torch.stack([item.construct_mask for item in processed_items]),
            'correct_options': torch.stack([item.correct_answer for item in processed_items]),
            'selected_options': torch.stack([item.answer_value for item in processed_items])
        }

    def collate_sequences(self, batch_tensors: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate multiple sequences into a single batch"""
        # Get sequence lengths
        L = [tensors['labels'].size(0) for tensors in batch_tensors]
        max_seq_len = max(L)
        
        # Create attention mask
        mask = torch.zeros(len(L), max_seq_len, dtype=torch.bool)
        for i, seq_len in enumerate(L):
            mask[i, :seq_len] = 1
        
        # Pad all sequences
        return {
            'question_ids': pad_sequence([b['questionids'] for b in batch_tensors], batch_first=True),
            'labels': pad_sequence([b['labels'] for b in batch_tensors], batch_first=True),
            'time_steps': pad_sequence([b['time_steps'] for b in batch_tensors], batch_first=True),
            'QuestionEmbedding': pad_sequence([b['QuestionEmbedding'] for b in batch_tensors], batch_first=True),
            'construct_ids': pad_sequence([b['construct_ids'] for b in batch_tensors], batch_first=True),
            'construct_mask': pad_sequence([b['construct_mask'] for b in batch_tensors], batch_first=True),
            'correct_options': pad_sequence([b['correct_options'] for b in batch_tensors], batch_first=True),
            'selected_options': pad_sequence([b['selected_options'] for b in batch_tensors], batch_first=True),
            'L': torch.tensor(L, dtype=torch.int64),
            'mask': mask
        }

def options_dataloader_preproc_process(batch):
    """Main preprocessing function for the dataloader"""
    preprocessor = SequencePreprocessor()
    
    # Process each sequence
    processed_sequences = [preprocessor.process_sequence(sequence) for sequence in batch]
    
    # Collate all sequences into a batch
    return preprocessor.collate_sequences(processed_sequences)