from typing import Iterable, Dict
import torch 
from torch import nn, Tensor
from .loss_rag import LossRAG
from ..constant import EMBEDDING_IN_BATCH_NEGATIVES

def cos_sim(a, b): 
    a_norm= torch.nn.functional.normalize(a, p= 2, dim= -1)
    b_norm= torch.nn.functional.normalize(b, p= 2, dim= -1)

    return torch.mm(a_norm, b_norm.transpose(0, 1))

class InBatchNegativeLoss(LossRAG): 
    def __init__(self, temp= 0.05): 
        super(InBatchNegativeLoss, self).__init__()
        self.temp= temp 
        self.loss_fct= nn.CrossEntropyLoss()

        self.pretty_name= "in_batch_negatives"
        self.task_name= EMBEDDING_IN_BATCH_NEGATIVES
    
    def _get_config_params(self):
        return {
            'tempurature': self.temp,
            'pretty_name': self.pretty_name, 
            'task_name': self.task_name
        }
    
    def forward(self,  features: Iterable[Dict[str, Tensor]]): 
        embeddings= self.model(features, return_embeddings= True)

        embedding_a= embeddings[0]
        embedding_b= torch.cat(embeddings[1:])

        cos_score= cos_sim(embedding_a, embedding_b) / self.temp # b x b 
        labels= torch.arange(len(cos_score), device= cos_score.device)

        return self.loss_fct(cos_score, labels)
    