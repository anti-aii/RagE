from typing import Union
import copy
import pandas as pd 
from torch.utils.data import Dataset 
from transformers import AutoTokenizer
from ..datareader import DataReader 

### only support for bloomz 
class GenAnsDL(Dataset): 
  
  def __init__(self, path_data_or_dataframe: Union[pd.DataFrame, str]):
    self.df= DataReader(path_data_or_dataframe).read()

  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, index): 
    x = self.df.iloc[index, 0] + "</s>"
    return x 

class GenAnsCollate: 
  def __init__(self, tokenizer_name: str= 'bigscience/tokenizer', max_length= 256):
    
    self.tokenizer= AutoTokenizer.from_pretrained(tokenizer_name, use_fast= True)
    self.max_length= max_length
    self.tokenizer.padding_side= 'left'
  
  def __call__(self, data): 
    x = self.tokenizer.batch_encode_plus(data, 
                                   truncation= True, 
                                   padding= 'longest',
                                   max_length= self.max_length, 
                                   return_tensors= 'pt')
    
    label = copy.deepcopy(x['input_ids'])
    label[label == self.tokenizer.pad_token_id] = -100 


    return {
          'x_ids': x.input_ids,
          'x_mask': x.attention_mask, 
          'label': label, 
      }

