import pandas as pd 
import torch 
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from ..datareader import DataReader 

class SentABDL(Dataset): 
    
    def __init__(self, path_data: str):
        self.df = DataReader(path_data).read()


    def __len__(self): 
        return len(self.df)

    def __getitem__(self, index): 
        sent1= self.df.iloc[index, 0] 
        sent2= self.df.iloc[index, 1]
        label= self.df.iloc[index, 2]

        return sent1, sent2, label 
    
class SentABCollate: 
    def __init__(self, tokenizer_name: str= 'vinai/phobert-base-v2',
                    mode: str= 'cross_encoder'):
        assert mode in ['bi_encoder', 'cross_encoder']
        self.tokenizer= AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space= True,
                                                      use_fast= True)
        self.mode= mode
    
    def _tokenize(self, list_text):
        x= self.tokenizer.batch_encode_plus(list_text, 
                            truncation= True, 
                            padding= 'longest',
                            return_tensors= 'pt', 
                            max_length= 256)
        
        return x

    def __call__(self, data): 
        sent1, sent2, label= zip(*data)
        if self.mode == 'cross_encoder': 
            text= list(zip(sent1, sent2))
            text= list(map(lambda x: self.tokenizer.sep_token.join(x), text))
            x = self._tokenize(text)
            return {
                'x': x, 
                'label': torch.tensor(label) 
            }
        elif self.mode == 'bi_encoder': 
            x1= self._tokenize(sent1)
            x2= self._tokenize(sent2)

            return {
                'x_1': x1,
                'x_2': x2, 
                'label': torch.tensor(label)
            }



    