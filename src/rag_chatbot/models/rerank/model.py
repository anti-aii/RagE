from typing import List
import torch 
import torch.nn as nn 
from transformers import AutoModel, AutoTokenizer
from ..componets import AttentionWithContext, ExtraRoberta 
from ...utils import TextFormat

### Cross-encoder
class CrossEncoder(nn.Module): 
    # using 
    def __init__(self, model_name= 'vinai/phobert-base-v2', required_grad= False, 
                 dropout= 0.1, hidden_dim= 768, num_label= 1):
        super(CrossEncoder, self).__init__()
        self.model= AutoModel.from_pretrained(model_name, output_hidden_states= True)

        if required_grad:
            self.model.requires_grad_(False)
    
        # define 
        self.extract= ExtraRoberta(method= 'mean')
        # only support transformer-based encoder output dim = 786
        self.attention_context= AttentionWithContext(units= hidden_dim, )
        # self.lnrom= nn.LayerNorm(hidden_dim, eps= 1e-6)

        # dropout 
        self.drp1= nn.Dropout(p= dropout)
        self.drp2= nn.Dropout(p= dropout)

        # defind output 
        self.fc= nn.Linear(hidden_dim, num_label)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def get_embedding(self, ids, mask):
        embedding_bert= self.model(ids, mask)
        embedding_enhance= self.extract(embedding_bert.hidden_states)

        # x= self.lnrom(embedding_enhance)
        x= self.drp1(embedding_enhance)
        x= self.attention_context(x)

        return x 
    
    
    def forward(self, ids, mask): 
        x= self.get_embedding(ids, mask)
        x= self.drp2(x)
        x= self.fc(x)

        return x 



### ReRanker 
class Reranker: 
    def __init__(self, model_name='vinai/phobert-base-v2', required_grad=False, 
                 dropout=0.1, hidden_dim=768, num_label=1):

        self.model= CrossEncoder(model_name, required_grad, dropout, hidden_dim, num_label)
        self.tokenizer= AutoTokenizer.from_pretrained(model_name, add_prefix_space= True, use_fast= True)

    def load_ckpt(self, path):
        self.model.load_state_dict(torch.load(path))

    def _preprocess(self):
        if self.model.training: 
            self.model.eval()
    
    def _preprocess_tokenize(self, text): 
        inputs= self.tokenizer.batch_encode_plus(text, return_tensors= 'pt', 
                            padding= 'longest', max_length= 256, truncation= True)
        
        return inputs

    def encode(self, text: List[list[str]]): 
        self._preprocess()
        batch_text= list(map(lambda x: self.tokenizer.sep_token.join(x), text))
        inputs= self._preprocess_tokenize(batch_text)

        embedding= []

        with torch.no_grad(): 
            for idx, data in enumerate(inputs): 
                embedding.append(self.model.get_embedding(data['input_ids'], 
                                                    data['attention_mask']))
        
        return embedding 

    
    







        