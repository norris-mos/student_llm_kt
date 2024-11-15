import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Union, NamedTuple
from datetime import datetime
import ast

class SequenceItem(NamedTuple):
    """Represents a single item in a sequence with its tensors"""
    question_id: torch.Tensor
    correctness: torch.Tensor
    time_features: torch.Tensor
    answer_value: torch.Tensor
    correct_answer: torch.Tensor
    question_embedding: torch.Tensor
    process_embedding: torch.Tensor
    process_ids: torch.Tensor
    process_mask: torch.Tensor
    misconception_ids: torch.Tensor
    misconception_mask: torch.Tensor
    subject_ids: torch.Tensor
    subject_mask: torch.Tensor

class SequencePreprocessor:
    def __init__(self):
        pass
    
    @staticmethod
    def create_cosine_features(item) -> torch.Tensor:
        # Your existing create_cosine_features implementation
        pass

    def process_single_item(self, item, max_miscon: int, max_subject_id: int, max_process: int) -> SequenceItem:
        """Process a single item in the sequence"""
        # Basic features
        correctness = torch.tensor(float(item.isCorrect), dtype=torch.float)
        question_id = torch.tensor(item.questionId, dtype=torch.long)
        answer_value = torch.tensor(item.answer_value - 1, dtype=torch.long)
        correct_answer = torch.tensor(item.correct_answer - 1, dtype=torch.long)
        
        # Time features
        cosine_time, cosine_day = self.create_cosine_features(item)
        time_features = torch.cat([cosine_time, cosine_day], dim=1)
        
        # Embeddings
        q_embed = torch.tensor(ast.literal_eval(item.question_embedding))
        p_embed = torch.tensor(ast.literal_eval(item.process_embedding))
        
        # Process tags with padding
        process_tags = torch.tensor(item.process, dtype=torch.long)
        process_mask = torch.ones(len(process_tags), dtype=torch.bool)
        process_ids = F.pad(process_tags, (0, max_process - len(process_tags)), value=0)
        process_mask = F.pad(process_mask, (0, max_process - len(process_mask)), value=False)
        
        # Misconception IDs with padding
        miscon_ids = torch.tensor(item.q_misconceptions, dtype=torch.long)
        miscon_mask = torch.ones(len(miscon_ids), dtype=torch.bool)
        miscon_ids = F.pad(miscon_ids, (0, max_miscon - len(miscon_ids)), value=0)
        miscon_mask = F.pad(miscon_mask, (0, max_miscon - len(miscon_mask)), value=False)
        
        # Subject IDs with padding
        subject_ids = torch.tensor(item.subjectId, dtype=torch.long)
        subject_mask = torch.ones(len(subject_ids), dtype=torch.bool)
        subject_ids = F.pad(subject_ids, (0, max_subject_id - len(subject_ids)), value=0)
        subject_mask = F.pad(subject_mask, (0, max_subject_id - len(subject_mask)), value=False)
        
        return SequenceItem(
            question_id=question_id,
            correctness=correctness,
            time_features=time_features,
            answer_value=answer_value,
            correct_answer=correct_answer,
            question_embedding=q_embed,
            process_embedding=p_embed,
            process_ids=process_ids,
            process_mask=process_mask,
            misconception_ids=miscon_ids,
            misconception_mask=miscon_mask,
            subject_ids=subject_ids,
            subject_mask=subject_mask
        )

    def process_sequence(self, sequence: List) -> Dict[str, torch.Tensor]:
        """Process a single sequence of items"""
        # Sort sequence by timestamp
        sorted_items = sorted(sequence, key=lambda x: datetime.strptime(x.time, '%Y-%m-%d %H:%M:%S.%f'))
        
        # Get max lengths for padding
        max_miscon = max(len(item.q_misconceptions) for item in sorted_items)
        max_subject_id = max(len(item.subjectId) for item in sorted_items)
        max_process = max(len(item.process) for item in sorted_items)
        
        # Process each item
        processed_items = [
            self.process_single_item(item, max_miscon, max_subject_id, max_process)
            for item in sorted_items
        ]
        
        # Stack tensors
        return {
            'questionids': torch.stack([item.question_id for item in processed_items]),
            'labels': torch.stack([item.correctness for item in processed_items]),
            'time_steps': torch.stack([item.time_features for item in processed_items]),
            'QuestionEmbedding': torch.stack([item.question_embedding for item in processed_items]),
            'processEmbedding': torch.stack([item.process_embedding for item in processed_items]),
            'misconceptions': torch.stack([item.misconception_ids for item in processed_items]),
            'misconception_mask': torch.stack([item.misconception_mask for item in processed_items]),
            'correct_options': torch.stack([item.correct_answer for item in processed_items]),
            'selected_options': torch.stack([item.answer_value for item in processed_items]),
            'subject_ids': torch.stack([item.subject_ids for item in processed_items]),
            'subject_ids_mask': torch.stack([item.subject_mask for item in processed_items]),
            'process_ids': torch.stack([item.process_ids for item in processed_items]),
            'process_id_mask': torch.stack([item.process_mask for item in processed_items])
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
            'questionids': pad_sequence([b['questionids'] for b in batch_tensors], batch_first=True),
            'labels': pad_sequence([b['labels'] for b in batch_tensors], batch_first=True),
            'time_steps': pad_sequence([b['time_steps'] for b in batch_tensors], batch_first=True),
            'QuestionEmbedding': pad_sequence([b['QuestionEmbedding'] for b in batch_tensors], batch_first=True),
            'processEmbedding': pad_sequence([b['processEmbedding'] for b in batch_tensors], batch_first=True),
            'misconceptions': pad_sequence([b['misconceptions'] for b in batch_tensors], batch_first=True),
            'misconception_mask': pad_sequence([b['misconception_mask'] for b in batch_tensors], batch_first=True),
            'correct_options': pad_sequence([b['correct_options'] for b in batch_tensors], batch_first=True),
            'selected_options': pad_sequence([b['selected_options'] for b in batch_tensors], batch_first=True),
            'subject_ids': pad_sequence([b['subject_ids'] for b in batch_tensors], batch_first=True),
            'subject_ids_mask': pad_sequence([b['subject_ids_mask'] for b in batch_tensors], batch_first=True),
            'process_ids': pad_sequence([b['process_ids'] for b in batch_tensors], batch_first=True),
            'process_id_mask': pad_sequence([b['process_id_mask'] for b in batch_tensors], batch_first=True),
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