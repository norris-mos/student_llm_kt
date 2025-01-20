import sys
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Process-Knowledge-Tracing
data_dir = os.path.dirname(project_root)  # Go up one more level to get to data



script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Process-Knowledge-Tracing
data_dir = os.path.dirname(project_root)  # Go up one more level to get to datacvc
print(script_dir)
print(project_root)
print(data_dir)


# Add source directories to path
sys.path.append('/mnt/ceph_rbd/robustness/')
sys.path.append('/mnt/ceph_rbd/student_llm_kt/src/')
sys.path.append(os.path.join(script_dir, 'src/DKT_src'))

from transformers import AutoTokenizer
from preproc_robust import DataFrame2InteractionDictionaryRobust, load_data
from LoRa_preprocessing import StudentInteractionsDataset
from dataloader_new import SequenceDataset, options_dataloader_preproc_process

ood_questions = [ 555,  495,  439,  837,  562,  696,  322,   77,  200,  477,  511,
        448,  134,  652,  767,  507,  378, 1000,  274, 1317,  930, 1347,
        489, 1330,  846,  114, 1095,  855,  783,  840,  872, 1533,  888,
       1176,  830,  962,  277, 1080,  905,  795,  789, 1087,  364,  804,
       1151]

# Load data using path to data directory
answers, questions, misconceptions, question_subject = load_data(os.path.join(data_dir, 'data'))

data = DataFrame2InteractionDictionaryRobust(answers, questions, misconceptions, question_subject, train_split=0.9)
data.createTrainDictionaryOOD(577575757,OOD=ood_questions)
data.createTestDictOOD(questions=ood_questions)


tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-bnb-4bit", 
    padding_side="right",
    truncation_side="right"
)

# finetune cache
train_cache = os.path.join(data_dir, 'data/interaction_dictionaries/filtered_interaction_dictionaries/train_ood_0.9.pt')
test_cache = os.path.join(data_dir, 'data/interaction_dictionaries/filtered_interaction_dictionaries/test_ood_0.1.pt')
train = StudentInteractionsDataset(data.train_dictionary, tokenizer, 4848484848, cache_path=train_cache)
test = StudentInteractionsDataset(data.test_dictionary, tokenizer, 8498578498, cache_path=test_cache)

