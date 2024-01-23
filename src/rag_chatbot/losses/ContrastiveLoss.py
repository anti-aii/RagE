import torch 
from torch import nn 

## Idea from OnlineContrastiveLoss of sentence-transformers 



class ContrastiveLoss(nn.Module):
    def __init__(self, margin): 
        super(ContrastiveLoss, self).__init__()
        self.margin= margin
        self.distance= lambda x, y: 1- torch.cosine_similarity(x, y)

    def forward(self, embedding_a, embedding_b, labels): 

        distance_matrix= self.distance(embedding_a, embedding_b)
        negs= distance_matrix[labels == 0]
        poss= distance_matrix[labels == 1]
        hard_negative = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
        hard_positive= poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

        positive_loss = hard_negative.pow(2).sum()
        negative_loss = nn.functional.relu(self.margin - hard_positive).pow(2).sum()
        loss = positive_loss + negative_loss
        return loss
