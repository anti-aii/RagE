import torch 
from torch import nn

class MSELogLoss(nn.Module): 
    def __init__(self, loss_fct= nn.MSELoss(), activation_fct= nn.Sigmoid(), cos_score_transformation= nn.Identity()): 
        super(MSELogLoss, self).__init__()
        self.loss_fct= loss_fct
        self.activation_fct= activation_fct 
        self.cos_score_transformation= cos_score_transformation

    def forward(self, embedding_sentence, labels): 
        output= self.cos_score_transformation(self.activation_fct(embedding_sentence))
        return self.loss_fct(output, labels.view(-1))