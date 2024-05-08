from typing import Dict, Iterable
import torch 
from torch import nn, Tensor
from .loss_rag import LossRAG
from ..constant import EMBEDDING_CONTRASTIVE


class ContrastiveLoss(LossRAG):
    def __init__(self, margin: float= 0.5): 
        super(ContrastiveLoss, self).__init__()
        self.margin= margin
        self.distance= lambda x, y: 1- torch.cosine_similarity(x, y)

        self.pretty_name= "contrastive"
        self.task_name= EMBEDDING_CONTRASTIVE

    def _get_config_params(self):
        return {
            'margin': self.margin, 
            'pretty_name': self.pretty_name, 
            'task_name': self.task_name
        }

    def forward(self, features: Iterable[Dict[str, Tensor]], labels: Tensor): 
        embeddings= self.model(features, return_embeddings= True)

        distance_matrix= self.distance(embeddings[0], embeddings[1])
        negs= distance_matrix[labels == 0]
        poss= distance_matrix[labels == 1]
        hard_negative = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
        hard_positive= poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

        positive_loss = hard_negative.pow(2).sum()
        negative_loss = nn.functional.relu(self.margin - hard_positive).pow(2).sum()
        loss = positive_loss + negative_loss
        return loss
