import torch.nn as nn 
from transformers import AutoModel
from ..componets import AttentionWithContext, ExtraRoberta 


class ReRanker(nn.Module): 
    # using 
    def __init__(self, model_name= 'vinai/phobert-base-v2', required_grad= False, 
                 dropout= 0.3, hidden_dim= 768, num_label= 1):
        super(ReRanker, self).__init__()
        self.model= AutoModel.from_pretrained(model_name, output_hidden_states= True)
        if required_grad:
            self.model.requires_grad_(False)
    
        # define 
        self.extract= ExtraRoberta(method= 'mean')
        # only support transformer-based encoder output dim = 786
        self.attention_context= AttentionWithContext(units= hidden_dim, )
        self.lnrom= nn.LayerNorm(hidden_dim, eps= 1e-6)

        # dropout 
        self.dropout= nn.Dropout(p= dropout)

        # defind output 
        self.fc= nn.Linear(hidden_dim, num_label)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, ids, mask): 
        embedding_bert= self.model(ids, mask)
        embedding_enhance= self.extract(embedding_bert.hidden_states)

        x= self.lnrom(embedding_enhance)
        x= self.dropout(x)
        x= self.fc(x)





        