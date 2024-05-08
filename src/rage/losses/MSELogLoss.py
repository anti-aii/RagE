from typing import Iterable, Dict
import torch 
from torch import nn, Tensor

from .loss_rag import LossRAG
from ..constant import EMBEDDING_RANKER_NUMERICAL

class MSELogLoss(LossRAG): 
    def __init__(self, loss_fct= nn.MSELoss(), activation_fct= nn.Sigmoid(), cos_score_transformation= nn.Identity()): 
        super(MSELogLoss, self).__init__()
        self.loss_fct= loss_fct
        self.activation_fct= activation_fct 
        self.cos_score_transformation= cos_score_transformation

        self.task_name= EMBEDDING_RANKER_NUMERICAL
        self.pretty_name= "mselog"

    def _get_config_params(self):
        return {
            'pretty_name': self.pretty_name, 
            'task_name': self.task_name
        }

    def forward(self, features: Iterable[Dict[str, Tensor]], labels: Tensor): 
        output= self.model(features)

        output= self.cos_score_transformation(self.activation_fct(output))
        return self.loss_fct(output.view(-1,), labels.view(-1).to(dtype= torch.float32))