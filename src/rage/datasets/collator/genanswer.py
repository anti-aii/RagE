from typing import Union, Type
import copy
import pandas as pd 
from torch.utils.data import Dataset 
import datasets
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from ..datareader import DataReader 

### only support for bloomz 
class GenAnsDL(Dataset): 
  
  def __init__(self, path_data_or_dataframe: Union[pd.DataFrame, str, datasets.Dataset]):
    self.data= DataReader(path_data_or_dataframe).read()

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, index): 
    if isinstance(self.data, datasets.Dataset): 
      return self.data[index].values()
    else: 
      x = self.df.iloc[index, 0].tolist() 
    return x 

class GenAnsCollate: 
  def __init__(
    self, 
    tokenizer: Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast]= 'bigscience/tokenizer', 
    max_length= 256,
    advance_config_encode: Type[dict]= None,
  ):

    if isinstance(tokenizer, str):
        self.tokenizer= AutoTokenizer.from_pretrained(tokenizer, use_fast= True)
    elif isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast): 
        self.tokenizer= tokenizer 

    self.max_length= max_length
    if advance_config_encode is None: 
      self.advance_config_encode= dict()
    else:
      self.advance_config_encode= advance_config_encode
      
    self.tokenizer.padding_side= 'left'
  
  def __call__(self, data): 
    data= list(map(lambda x: x + self.tokenizer.eos_token, data))

    x = self.tokenizer.batch_encode_plus(data, 
                                   truncation= True, 
                                   padding= 'longest',
                                   max_length= self.max_length, 
                                   return_tensors= 'pt',
                                   **self.advance_config_encode)
    
    label = copy.deepcopy(x['input_ids'])
    label[label == self.tokenizer.pad_token_id] = -100 


    return {
          'x_ids': x.input_ids,
          'x_mask': x.attention_mask, 
          'label': label, 
      }

