from typing import Iterable, Dict, Union
import torch 
from torch import nn, Tensor
from .loss_rag import LossRAG
from ..constant import EMBEDDING_RANKER_NUMERICAL



class BinaryCrossEntropy(LossRAG): 
    def __init__(
        self, 
        weight: Tensor = None, 
        size_average = None, 
        reduce = None, 
        reduction: str = 'mean', 
        pos_weight: Tensor = None):
        
        super(BinaryCrossEntropy, self).__init__()

        self.loss_fct= nn.BCEWithLogitsLoss(
            weight, 
            size_average,
            reduce,
            reduction, 
            pos_weight 
        )
        
        self.weight= weight
        self.size_avg= size_average 
        self.reduce= reduce 
        self.reduction= reduction
        self.pos_weight= pos_weight

        self.pretty_name= "binary_crossentropy"
        self.task_name= EMBEDDING_RANKER_NUMERICAL

    def _get_config_params(self):
        return {
            'weight': self.weight, 
            'size_avg': self.size_avg, 
            'reduce': self.reduce, 
            'reduction': self.reduction, 
            'pos_weight': self.pos_weight, 
            'pretty_name': self.pretty_name, 
            'task_name': self.task_name
        }
    
    def forward(
        self, 
        features: Union[Iterable[Dict[str, Tensor]], Dict[str, Tensor]], 
        labels: Tensor
    ): 
        output= self.model(features)

        return self.loss_fct(output.view(-1,), labels.view(-1,).to(dtype= torch.float32))
