import torch 
import torch.nn as nn 
from .attention_context import AttentionWithContext 


class PoolingStrategy(nn.Module): 
    def __init__(self, strategy: str, units: int= 256):
        super(PoolingStrategy, self).__init__() 
        if strategy not in ["max", "mean", "first", "attention_context", 
                            "dense_first", "dense_avg", "dense_max"]:
            raise ValueError("max, mean, first, attention_context, dense_first, dense_avg, dense_max")
        
        if strategy == "attention_context": 
            self.learn_params= AttentionWithContext(units= units)

        if strategy in ["dense_avg", "dense_first", "dense_max"]: 
            self.learn_params= nn.Linear(units, units)

        self.strategy= strategy 

    def forward(self, hidden_states): 
        if self.strategy == "max" or self.strategy == "dense_max": 
            embedding= torch.max(hidden_states, dim= 1)
        if self.strategy == "mean" or self.strategy == "dense_avg": 
            embedding= torch.mean(hidden_states, dim= 1)
        if self.strategy == "first" or self.strategy == "dense_first": 
            embedding= hidden_states[:, 0, :]

        if self.strategy == "attention_context": 
            embedding= self.learn_params(hidden_states)

        if self.strategy in ["dense_avg", "dense_first", "dense_max"]: 
            embedding= self.learn_params(embedding)

        return embedding        
