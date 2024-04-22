import torch 
import torch.nn as nn 

class TripletLoss(nn.Module):
    def __init__(self, margin= 0.5):
        super(TripletLoss, self).__init__() 
        self.margin= margin 
        self.distance= lambda x, y: 1 - torch.cosine_similarity(x, y)

    def forward(self, embedding_anchor, embedding_pos, embedding_neg): 
        distance_pos= self.distance(embedding_anchor, embedding_pos)
        distance_neg= self.distance(embedding_anchor, embedding_neg)

        return nn.relu(distance_pos - distance_neg + self.margin).mean()
    