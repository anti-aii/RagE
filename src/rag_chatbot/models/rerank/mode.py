import torch 
import torch.nn as nn 
from transformers import AutoTokenizer, AutoModel

class ReRank(nn.Module): 
    def __init__(self, model_name= 'vinai/phobert-base-v2'):
        self.model= AutoModel.from_pretrained(model_name)
        