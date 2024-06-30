from typing import Iterable, Dict
import torch 
from torch import nn, Tensor
from .loss_rag import LossRAG
from ..constant import EMBEDDING_RANKER_NUMERICAL


class CategoricalCrossEntropy(LossRAG): 
    def __init__(
        self, 
        weight: Tensor = None, 
        size_average = None, 
        reduce = None, 
        reduction: str = 'mean', 
        pos_weight: Tensor = None,
        label_smoothing: float= 0.
        ):

        super(CategoricalCrossEntropy, self).__init__() 
        
        self.loss_fct= nn.CrossEntropyLoss(
            weight= weight, 
            size_average= size_average, 
            reduce= reduce, 
            reduction= reduction, 
            label_smoothing= label_smoothing
        )
        
        self.weight= weight
        self.size_avg= size_average 
        self.reduce= reduce 
        self.reduction= reduction
        self.pos_weight= pos_weight
        self.label_smoothing= label_smoothing

        self.pretty_name= "categorical_crossentropy"
        self.task_name= EMBEDDING_RANKER_NUMERICAL

    def _get_config_params(self):
        return {
            'weight': self.weight, 
            'size_avg': self.size_avg, 
            'reduce': self.reduce, 
            'reduction': self.reduction, 
            'pos_weight': self.pos_weight, 
            'label_smoothing': self.label_smoothing, 
            'pretty_name': self.pretty_name, 
            'task_name': self.task_name
        }
    
    def forward(self, features: Iterable[Dict[str, Tensor]], labels: Tensor): 
        output= self.model(features)
        return self.loss_fct(output, labels.view(-1,))