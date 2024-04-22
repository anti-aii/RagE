import torch 
from torch import nn

class CosineSimilarityLoss(nn.Module): 
    def __init__(self, loss_fct= nn.MSELoss(), cos_score_transformation= nn.Identity()): 
        super(CosineSimilarityLoss, self).__init__()
        self.loss_fct= loss_fct
        self.cos_score_transformation= cos_score_transformation

    def forward(self, embedding_a, embedding_b, labels): 
        output= self.cos_score_transformation(torch.cosine_similarity(embedding_a, embedding_b))
        return self.loss_fct(output, labels.view(-1))