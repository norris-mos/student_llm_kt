import pandas as pd
import sys
sys.path.append('/mnt/ceph_rbd/LoRa/student_llm_kt/src/DKT_src')
from datasets import Dataset
import os
from collections import defaultdict , namedtuple
from prompts import PROMPT_TEMPLATE, INSTRUCTION, PROMPT_TEMPLATE_TEST, STUDENT_START_TOKEN, INTERACTION_SEP_TOKEN, HISTORY_END_TOKEN
import math
import torch

from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
import numpy as np
from data_classes import EediItem



def load_data(path):



  data_path = os.path.join(path)
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






# Data preprocesssing to split users by history length and 

class DataFrame2InteractionDictionary():
    def __init__(self, answers, questions, misconceptions, question_subject, train_split=0.8, random_seed=42):
        self.user_ids = list(answers['UserId'].unique())
        self.answers = answers.drop('QuizSessionId',axis=1)
        self.questions = questions
        
        self.misconceptions = misconceptions
        self.question_subject = question_subject
        self.train_dictionary = defaultdict()
        self.test_dictionary = defaultdict()
        
        self.merge = self.answers.merge(questions, on='QuestionId', how='left').dropna().reset_index(drop=True)
        print(len(self.merge['QuestionId'].unique()))
        self.merge['QuestionId_mapped'], self.question_uniques = pd.factorize(self.merge['QuestionId'])
        self.merge['QuizId_mapped'], self.quiz_uniques = pd.factorize(self.merge['QuizId'])
        
        # Add 1 to shift from 0-based to 1-based indexing if needed
        self.merge['QuestionId_mapped'] = self.merge['QuestionId_mapped'] + 1
        self.merge['QuizId_mapped'] = self.merge['QuizId_mapped'] + 1
        self.merge['QuestionId'] = self.merge['QuestionId_mapped']
        self.merge['QuizId'] = self.merge['QuizId_mapped']
        
        # Store the number of unique values for the feature config
        self.num_questions = len(self.question_uniques) + 1  # +1 for 1-based indexing
        self.num_quizzes = len(self.quiz_uniques) + 1  # +1 for 1-based indexing


    
        self.train_split = train_split
        
        # Shuffle users with random seed
        np.random.seed(random_seed)
        shuffled_users = np.random.permutation(self.user_ids)
        
        # Split users into train and test
        split_idx = int(len(shuffled_users) * train_split)
        self.train_users = shuffled_users[:split_idx]
        self.test_users = shuffled_users[split_idx:]
        
        print(f"Total users: {len(self.user_ids)}")
        print(f"Train users: {len(self.train_users)} ({train_split*100:.1f}%)")
        print(f"Test users: {len(self.test_users)} ({(1-train_split)*100:.1f}%)")
        print(f"Number of QuestionIds is {self.num_questions}")
        print(f"Number of quizzes is {self.num_quizzes}")


    def createTestDict(self):
        
        """

        For testing we need to try and predict on each subset of context and hence have to split up each user

        Parameters:
        sort_by:
        - Creates the interaction dictionary with 
        
        """

        # loop over each users history
        interaction_counter = 0
        for user in self.test_users:

            # extract interaction df
            interactions = self.merge[self.merge['UserId']==user]


            sorted_interactions = interactions.sort_values(by='DateAnswered').reset_index(drop=True)


            history_cache = self.format_history_cached(sorted_interactions,include_fields=None)


            for index,row in sorted_interactions.iterrows():
                 
                 interaction_counter+=1
               
             
                 history = "\n\n".join(history_cache[:index])
              
                 
                 
      
                 self.test_dictionary[interaction_counter] = {'history':history,
                                                                    'question':row.QuestionText,
                                                                    'options':{'A':row.AnswerAText,'B':row.AnswerBText,'C':row.AnswerCText,'D':row.AnswerDText},
                                                                    'correct_answer':self.num2option(row.AnswerValue)
                                    }
                 
    def createTrainDictionary(self,max_seq_len):
        
        """
        Because during training we can use the whole context of all interactions as the finetuning material we can quickly load this.
        Parameters:
        sort_by:
        - Creates the interaction dictionary with 
        
        """

        # loop over each users history
        interaction_counter = 0
        for user in self.train_users:

            # extract interaction df
            interactions = self.merge[self.merge['UserId']==user]


            sorted_interactions = interactions.sort_values(by='DateAnswered').reset_index(drop=True)


            history_cache = self.format_history_cached(sorted_interactions,include_fields=None)


            if len(history_cache) <1:
                continue
                 
            interaction_counter+=1
        
        # take the biggest interaction length for this student if < max
            if len(history_cache)<=max_seq_len:
          
                history = "\n\n".join(history_cache[:-1])      
                self.train_dictionary[interaction_counter] = {'history':history,
                                                                'question':sorted_interactions.iloc[-1].QuestionText,
                                                              'options':{'A':sorted_interactions.iloc[-1].AnswerAText,'B':sorted_interactions.iloc[-1].AnswerBText,'C':sorted_interactions.iloc[-1].AnswerCText,'D':sorted_interactions.iloc[-1].AnswerDText},
                                                                'correct_answer':self.num2option(sorted_interactions.iloc[-1].AnswerValue)}
            else:
        
                history = "\n\n".join(history_cache[:max_seq_len]) 
               

                self.interactionDictionary[interaction_counter] = {'history':history,
                                                                'question':sorted_interactions.iloc[max_seq_len].QuestionText,
                                                                'options':{'A':sorted_interactions.iloc[max_seq_len].AnswerAText,'B':sorted_interactions.iloc[max_seq_len].AnswerBText,'C':sorted_interactions.iloc[max_seq_len].AnswerCText,'D':sorted_interactions.iloc[max_seq_len].AnswerDText},
                                                                'correct_answer':self.num2option(sorted_interactions.iloc[max_seq_len].AnswerValue)
                                }

    def createedi(self,max_seq_len,include_fields=None,cache_path='/mnt/ceph_rbd/LoRa/filtered_interaction_dictionaries/'):
        
        """
        Because during training we can use the whole context of all interactions as the finetuning material we can quickly load this.
        Parameters:
        sort_by:
        - Creates the interaction dictionary with 
        
        """

        # loop over each users history
        interaction_counter = 0
        for user in self.train_users:

            # extract interaction df
            interactions = self.merge[self.merge['UserId']==user]


            sorted_interactions = interactions.sort_values(by='DateAnswered').reset_index(drop=True)


            history_cache = self.eedi_format(sorted_interactions,include_fields)


            if len(history_cache) <1:
                continue
                 
            
        
        # take the biggest interaction length for this student if < max
            if len(history_cache)<=max_seq_len:
          
                  
                self.train_dictionary[interaction_counter] = history_cache
                interaction_counter+=1
                    # Save to cache if path provided
        if cache_path:
            train = cache_path + f"train_{self.train_split}.pt"
            test = cache_path + f"test_{1-self.train_split}.pt"
            print(f"Saving filtered dataset to cache: {train}")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(self.train_dictionary, train)
            torch.save(self.test_dictionary,test)

    def num2option(self,number):
        map_dict = {
            1:'A',
            2:'B',
            3:'C',
            4:'D'
        }
     
       
        option = map_dict[number]
        
        

        return option
    
        
    def format_history_cached(self, user_history_df, include_fields=None):
        if include_fields is None:
            include_fields = {
                'DATE-TIME': 'DateAnswered',
                'QUESTIONID': 'QuestionId',
                'QUESTION': 'QuestionText',
                'CONSTRUCT_NAME': 'ConstructName',
                'OPTIONS': ['AnswerAText', 'AnswerBText', 'AnswerCText', 'AnswerDText'],
                'USER_ANSWER': 'AnswerValue',
                'CORRECT_ANSWER': 'CorrectAnswer',
                'QUESTION_CORRECT': 'IsCorrect'
            }

        history_string = []
        for _, interaction in user_history_df.iterrows():
      
            item_parts = []
            for label, field in include_fields.items():
                if isinstance(field, list):
                    # Handle special case for OPTIONS
                    options = [f"{chr(65+i)}={interaction[f]}" for i, f in enumerate(field) if f in interaction]
                    item_parts.append(f"{label}: {' '.join(options)}")
                elif field in interaction:
                    value = interaction[field]
                    if field in ['AnswerValue', 'CorrectAnswer']:
                    
                        value = self.num2option(value)
                    item_parts.append(f"{label}: {value}")

            
            item_string = "\n".join(item_parts)

            history_string.append(item_string)
        
        return history_string

            
        
    def eedi_format(self, user_history_df, include_fields=None):

        if include_fields is None:


            include_fields = {
           
              'USER_ID': 'UserId',
              'QUESTION_ID': 'QuestionId',
              'QUIZ_ID': 'QuizId',
              'IS_CORRECT': 'IsCorrect',
              'ANSWER_VALUE': 'AnswerValue',
              'DATE_ANSWERED': 'DateAnswered',
              'ANSWER_TYPE': 'AnswerType',
              'CONSTRUCT_ID': 'ConstructId',
              'CONSTRUCT_NAME': 'ConstructName',
              'CORRECT_ANSWER': 'CorrectAnswer',
              'EXPLANATION_A': 'ExplanationA',
              'EXPLANATION_B': 'ExplanationB',
              'EXPLANATION_C': 'ExplanationC',
              'EXPLANATION_D': 'ExplanationD',
              'QUESTION_TEXT': 'QuestionText',
              'ANSWER_A_TEXT': 'AnswerAText',
              'ANSWER_B_TEXT': 'AnswerBText',
              'ANSWER_C_TEXT': 'AnswerCText',
              'ANSWER_D_TEXT': 'AnswerDText',
         
          }
                    
        

        item_history = []
        for _, interaction in user_history_df.iterrows():
            
            item = EediItem.from_interaction(interaction)
            item_history.append(item)





          


            
        return item_history
    
    def format_history(self, index, user_history_df, include_fields=None):
        if include_fields is None:
            include_fields = {
                'DATE-TIME': 'DateAnswered',
                'QUESTIONID': 'QuestionId',
                'QUESTION': 'QuestionText',
                'CONSTRUCT_NAME': 'ConstructName',
                'OPTIONS': ['AnswerAText', 'AnswerBText', 'AnswerCText', 'AnswerDText'],
                'USER_ANSWER': 'AnswerValue',
                'CORRECT_ANSWER': 'CorrectAnswer',
                'QUESTION_CORRECT': 'IsCorrect'
            }

        history_string = []
        for _, interaction in user_history_df.iloc[:index].iterrows():
      
            item_parts = []
            for label, field in include_fields.items():
                if isinstance(field, list):
                    # Handle special case for OPTIONS
                    options = [f"{chr(65+i)}={interaction[f]}" for i, f in enumerate(field) if f in interaction]
                    item_parts.append(f"{label}: {' '.join(options)}")
                elif field in interaction:
                    value = interaction[field]
                    if field in ['AnswerValue', 'CorrectAnswer']:
                    
                        value = self.num2option(value)
                    item_parts.append(f"{label}: {value}")
            
            item_string = "\n".join(item_parts)
            history_string.append(item_string)
        
        return "\n\n".join(history_string)

            



        


class StudentInteractionsDataset(Dataset):
    def __init__(self, data_dict, tokenizer, max_length, cache_path=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Try to load from cache first
        if cache_path and os.path.exists(cache_path):
            print(f"Loading filtered dataset from cache: {cache_path}")
            self._data = torch.load(cache_path)
        else:
            print("Filtering long sequences...")
            # Filter out examples that are too long
            filtered_data = {}
            total = len(data_dict)
            for i, (key, item) in enumerate(data_dict.items()):
                if i % 1000 == 0:  # Progress update
                    print(f"Processing {i}/{total} items...")
                    
                prompt = PROMPT_TEMPLATE.format(
                    INSTRUCTION=INSTRUCTION,
                    history=item['history'],
                    question=item['question'],
                    option_a=item['options']['A'],
                    option_b=item['options']['B'],
                    option_c=item['options']['C'],
                    option_d=item['options']['D'],
                    RESPONSE=item['correct_answer']
                )
                prompt = self.tokenizer.bos_token + prompt + self.tokenizer.eos_token
                if len(self.tokenizer.encode(prompt)) <= 40000:
                    filtered_data[key] = item
            
            self._data = filtered_data
            
            # Save to cache if path provided
            if cache_path:
                print(f"Saving filtered dataset to cache: {cache_path}")
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                torch.save(self._data, cache_path)
            
            print(f"Filtered dataset contains {len(self._data)} items (removed {total - len(self._data)} items)")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        item = self._data[idx]
        prompt = PROMPT_TEMPLATE.format(
            INSTRUCTION=INSTRUCTION,
            history=item['history'],
            question=item['question'],
            option_a=item['options']['A'],
            option_b=item['options']['B'],
            option_c=item['options']['C'],
            option_d=item['options']['D'],
            RESPONSE=item['correct_answer']
        )
        #prompt = self.tokenizer.bos_token + prompt + self.tokenizer.eos_token
        prompt =  prompt + self.tokenizer.eos_token
        return prompt

    def __getTestitem__(self, idx):
        item = self._data[idx]
        prompt = PROMPT_TEMPLATE_TEST.format(
            INSTRUCTION=INSTRUCTION,
            history=item['history'],
            question=item['question'],
            option_a=item['options']['A'],
            option_b=item['options']['B'],
            option_c=item['options']['C'],
            option_d=item['options']['D'],
        )
        response = item['correct_answer']
        prompt = prompt + " "
        return prompt, response

    def load_debug_data(self, num_examples):
        texts = []
        keys = list(self._data.keys())[:num_examples]
        for key in keys:
            prompt = self.__getitem__(key)
            texts.append(prompt)
        
        return HFDataset.from_dict({
            "text": texts
        })

    def load_data(self):
        """Load all data"""
        texts = []
        for key in self._data:
            prompt = self.__getitem__(key)
            texts.append({"text": prompt})
        
        print(f'Number of Examples: {len(texts)}')
        return HFDataset.from_dict({"text": [item["text"] for item in texts]})

    def load_test_data(self):
        texts = []
        responses = []
        
        keys = list(self._data.keys())
        
        for key in keys:
            prompt, response = self.__getTestitem__(key)
            texts.append(prompt)
            responses.append(response)
            
        return HFDataset.from_dict({
            "text": texts,
            "response": responses
        })

    def load_test_debug_data(self, num_examples):
        texts = []
        responses = []
        
        keys = list(self._data.keys())[:num_examples]
        
        for key in keys:
            prompt, response = self.__getTestitem__(key)
            texts.append(prompt)
            responses.append(response)
            
        return HFDataset.from_dict({
            "text": texts,
            "response": responses
        })