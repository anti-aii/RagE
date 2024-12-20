import pandas as pd 
import torch 
from  torch.utils.data import Dataset 
import datasets
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from ..datareader import DataReader
from ...utils.augment_text import TextAugment
from typing import Union, Type

from ...constant import (
    EMBEDDING_RANKER_NUMERICAL,
    EMBEDDING_CONTRASTIVE,
    EMBEDDING_TRIPLET,
    EMBEDDING_IN_BATCH_NEGATIVES
)

import logging 
logger= logging.Logger(__name__)

class is_df_relate_task:
    # check related task 
    def  __init__(self, task: str): 
        self.task= task 
    
    def __call__(self, data: Union[pd.DataFrame, datasets.Dataset]):
        columns= data.column_names if isinstance(data, datasets.Dataset) else data.columns

        if self.task == EMBEDDING_RANKER_NUMERICAL and len(columns) != 3: 
            raise ValueError('The dataset is required to have 3 columns while using with cosine_sim, sigmoid, categorical_crossentroy loss')
        elif self.task == EMBEDDING_CONTRASTIVE and len(columns) != 3: 
            raise ValueError('The dataset is required to have 3 columns while using with contrastive loss') 
        elif self.task == EMBEDDING_TRIPLET and len(columns) != 3: 
            raise ValueError('The dataset is required to have 3 columns while using triplet loss')
        # elif self.task != EMBEDDING_IN_BATCH_NEGATIVES: 
        #     raise ValueError('The dataset is required to have 1, 2 or 3 columns while using in-batch negatives loss')
        
        return data

class SentABDL(Dataset): 
    
    def __init__(self, path_or_data: Union[str, pd.DataFrame, datasets.Dataset], task: str):        
        assert task in [
            EMBEDDING_RANKER_NUMERICAL,  # cosine_sim, sigmoid, categorical 
            EMBEDDING_CONTRASTIVE, # contrastive loss  
            EMBEDDING_TRIPLET, # triplet loss 
            EMBEDDING_IN_BATCH_NEGATIVES # in-batch negatives  
        ]
        self.data= DataReader(path_or_data, condition_func= is_df_relate_task(task)).read()
        self.task= task

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, index): 
        if isinstance(self.data, datasets.Dataset): 
            return self.data[index].values()
        else:  # dataframe 
            return self.data.iloc[index, :]
    
class SentABCollate: 
    def __init__(
        self, 
        tokenizer: Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast], 
        max_length= 256, 
        advance_config_encode: Type[dict]= None,
        mode: str= 'cross_encoder', 
        task= None, 
        augment_func: TextAugment= None):
        
        assert isinstance(augment_func, TextAugment) or augment_func is None
        assert mode in ['bi_encoder', 'cross_encoder']
        assert task in [
            EMBEDDING_RANKER_NUMERICAL,  # cosine_sim, sigmoid, categorical 
            EMBEDDING_CONTRASTIVE, # contrastive loss  
            EMBEDDING_TRIPLET, # triplet loss 
            EMBEDDING_IN_BATCH_NEGATIVES # in-batch negatives    
        ]
        if isinstance(tokenizer, str):
            self.tokenizer= AutoTokenizer.from_pretrained(tokenizer, add_prefix_space= True,
                                                      use_fast= True)
        elif isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast): 
            self.tokenizer= tokenizer 
            
        if mode == 'cross_encoder' and task != EMBEDDING_RANKER_NUMERICAL: 
            raise ValueError('To use mode= "cross_encoder", you must use task= EMBEDDING_RANKER_NUMERICAL')
        
        self.mode= mode
        self.max_length= max_length
        self.augument_func= augment_func
        self.task= task
        
        if advance_config_encode is None: 
            self.advance_config_encode= dict()
        else: 
            self.advance_config_encode= advance_config_encode

    def _tokenize(self, list_text):
        x= self.tokenizer.batch_encode_plus(list_text, 
                            truncation= True, 
                            padding= 'longest',
                            return_tensors= 'pt', 
                            max_length= self.max_length,
                            **self.advance_config_encode)
        return {
            'input_ids': x['input_ids'], 
            'attention_mask': x['attention_mask']
        }
    
    def _using_augment(self, data):
        # using noise dropout prob = 0.1
        return (list(map(lambda x: self.augument_func(x), i)) for i in data)

    def _return_type_numerical(self, data):
        # support cosine_sim, sigmoid, categorical_crossentropy loss 
        sent1, sent2, label= zip(*data)

        if self.augument_func: 
            sent1, sent2= self._using_augment(data= (sent1, sent2))

        if self.mode == 'cross_encoder': 
            text= list(zip(sent1, sent2))
            text= list(map(lambda x: self.tokenizer.sep_token.join(x), text))
            return {
                'x': self._tokenize(text), 
                'label': torch.tensor(label) 
            }
        elif self.mode == 'bi_encoder': 
            return {
                'x_1': self._tokenize(sent1),
                'x_2': self._tokenize(sent2), 
                'label': torch.tensor(label)
            }
    
    def _return_type_contrastive(self, data): 
        # support contrastive loss 
        sent1, sent2, label= zip(*data)

        if self.augument_func: 
            sent1, sent2= self._using_augment((sent1, sent2))
        
        return {
            'sent1': self._tokenize(sent1), 
            'sent2': self._tokenize(sent2), 
            'label': torch.tensor(label)
        }
        
                
    def _return_type_triplet(self, data): 
        sent1, sent2, sent3= zip(*data)

        if self.augument_func: 
            sent1, sent2, sent3= self._using_augment((sent1, sent2, sent3))
        
        return {
            'anchor': self._tokenize(sent1),
            'pos': self._tokenize(sent2), 
            'neg': self._tokenize(sent3)
        }

    def _return_type_inbatch_negative(self, data): 
        zip_data= list(zip(*data))
        if len(zip_data) == 1: 
            anchor, pos= zip_data[0], zip_data[0]
        else: 
            anchor, pos= zip_data[0: 2]
        hard_neg= zip_data[2:] # support multiple hard negatives

        if self.augument_func: 
            anchor, pos= self._using_augment((anchor, pos))

        result= {
                'anchor': self._tokenize(anchor), 
                'pos': self._tokenize(pos)
        }

        if len(hard_neg) > 0: 
            if self.augument_func:
                hard_neg= list(self._using_augment(hard_neg))
            for i in range(len(hard_neg)): 
                result[f'hard_neg_{i+1}']= self._tokenize(hard_neg[i])
            return result
        else:
            return result
    
    def _choice_return(self): 
        return {
            EMBEDDING_RANKER_NUMERICAL: self._return_type_numerical,  
            EMBEDDING_CONTRASTIVE: self._return_type_contrastive,
            EMBEDDING_TRIPLET: self._return_type_triplet, 
            EMBEDDING_IN_BATCH_NEGATIVES: self._return_type_inbatch_negative  
        }[self.task]

    def __call__(self, data): 
        return self._choice_return()(data= data)



    