from typing import Iterable, Dict
import torch 
from torch import nn, Tensor

from .loss_rag import LossRAG
from ..constant import EMBEDDING_TRIPLET

class TripletLoss(LossRAG):
    def __init__(self, margin= 0.5):
        super(TripletLoss, self).__init__() 
        self.margin= margin 
        self.distance= lambda x, y: 1 - torch.cosine_similarity(x, y)

        self.pretty_name= "triplet"
        self.task_name= EMBEDDING_TRIPLET
        
    def _get_config_params(self):
        return {
            'margin': self.margin, 
            'pretty_name': self.pretty_name, 
            'task_name': self.task_name
        }
    
    def forward(self, features: Iterable[Dict[str, Tensor]]): 
        embeddings= self.model(features, return_embeddings= True)

        distance_pos= self.distance(embeddings[0], embeddings[1])
        distance_neg= self.distance(embeddings[0], embeddings[2])

        return nn.relu(distance_pos - distance_neg + self.margin).mean()
    