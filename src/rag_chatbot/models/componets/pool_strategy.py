import torch 
import torch.nn as nn 
from .attention_context import AttentionWithContext 


class PoolingStrategy(nn.Module): 
    def __init__(self, strategy: str, units: int= 256):
        super(PoolingStrategy, self).__init__() 
        if strategy not in ["max", "mean", "first", "attention_context"]:
            raise ValueError("max, mean, first, attention_context")
        
        if strategy: 
            self.learn_params= AttentionWithContext(units= units)
        self.strategy= strategy 

    def forward(self, hidden_states): 
        if self.strategy == "max": 
            return torch.max(hidden_states, dim= 1)
        if self.strategy == "mean": 
            return torch.mean(hidden_states, dim= 1)
        if self.strategy == "first": 
            return hidden_states[:, 0, :]
        if self.strategy == "attention_context": 
            return self.learn_params(hidden_states)
        
