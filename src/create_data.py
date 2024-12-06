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
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'src/DKT_src'))

from transformers import AutoTokenizer
from LoRa_preprocessing import StudentInteractionsDataset, DataFrame2InteractionDictionary, load_data
from dataloader_new import SequenceDataset, options_dataloader_preproc_process

# Load data using path to data directory
answers, questions, misconceptions, question_subject = load_data(os.path.join(data_dir, 'data'))

data = DataFrame2InteractionDictionary(answers, questions, misconceptions, question_subject, train_split=0.9)
data.createTrainDictionary(3456784)
data.createTestDict()

tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-bnb-4bit", 
    padding_side="right",
    truncation_side="right"
)

# finetune cache
train_cache = os.path.join(data_dir, 'data/interaction_dictionaries/filtered_interaction_dictionaries/train_lora_qwen_0.9.pt')
test_cache = os.path.join(data_dir, 'data/interaction_dictionaries/filtered_interaction_dictionaries/test_lora_qwen_0.1.pt')
train = StudentInteractionsDataset(data.train_dictionary, tokenizer, 4848484848, cache_path=train_cache)
test = StudentInteractionsDataset(data.test_dictionary, tokenizer, 8498578498, cache_path=test_cache)
#eedi cache
train_cache_eedi = os.path.join(data_dir, 'data/interaction_dictionaries/filtered_interaction_dictionaries/train_eedi_0.9.pt')
test_cache_eedi = os.path.join(data_dir, 'data/interaction_dictionaries/filtered_interaction_dictionaries/test_eedi_0.1.pt')
train_eedi = data.creatEediTrain(434343434,cache_path=train_cache_eedi)
test_eedi = data.creatEediTest(47575477455,cache_path=test_cache_eedi)
