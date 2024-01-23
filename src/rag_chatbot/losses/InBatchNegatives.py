import torch 
import torch.nn as nn 

def cos_sim(a, b): 
    a_norm= torch.nn.functional.normalize(a, p= 2, dim= -1)
    b_norm= torch.nn.functional.normalize(b, p= 2, dim= -1)

    return torch.mm(a_norm, b_norm.transpose(0, 1))

class InBatchNegativeLoss(nn.Module): 
    def __init__(self, temp= 0.05): 
        super(InBatchNegativeLoss, self).__init__()
        self.temp= temp 
        self.loss_fct= nn.CrossEntropyLoss()
    
    def forward(self, embeddings): 
        embedding_a= embeddings[0]
        embedding_b= torch.cat(embeddings[1:])

        cos_score= cos_sim(embedding_a, embedding_b) / self.temp # b x b 
        labels= torch.arange(len(cos_score))

        return self.loss_fct(cos_score, labels)
    