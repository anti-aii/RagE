from typing import List
import torch 
import torch.nn as nn 
from transformers import AutoTokenizer
from ..componets import ExtraRoberta, load_backbone, PoolingStrategy


### Cross-encoder
class CrossEncoder(nn.Module): 
    # using 
    def __init__(self, model_name= 'vinai/phobert-base-v2', type_backbone= 'bert',
                 using_hidden_states= True, required_grad= False, 
                 strategy_pooling= "attention_context", dropout= 0.1, hidden_dim= 768, num_label= 1):
        super(CrossEncoder, self).__init__()
        
        self.using_hidden_states= using_hidden_states
        self.strategy_pooling= strategy_pooling

        self.pooling= PoolingStrategy(strategy= strategy_pooling, units= hidden_dim)
        
        self.model= load_backbone(model_name, type_backbone= type_backbone, dropout= dropout,
                                using_hidden_states= using_hidden_states)

        if not required_grad:
            self.model.requires_grad_(False)
    
        # define 
        if self.using_hidden_states:
            self.extract= ExtraRoberta(method= 'mean')
        
        if strategy_pooling == "attention_context": 
            self.drp1= nn.Dropout(p= dropout)

        # dropout 
        self.dropout_embedding= nn.Dropout(p= dropout)

        # defind output 
        self.fc= nn.Linear(hidden_dim, num_label)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def get_embedding(self, inputs):
        embedding= self.model(**inputs)

        if self.using_hidden_states: 
            embedding= self.extract(embedding.hidden_states)
        else:
            embedding= embedding.last_hidden_state

        if self.strategy_pooling == "attention_context": 
            embedding= self.drp1(embedding)
        x= self.pooling(embedding)

        return x 
    
    
    def forward(self, inputs): 
        x= self.get_embedding(inputs)
        x= self.dropout_embedding(x)
        x= self.fc(x)

        return x 



### ReRanker 
class Ranker: 
    def __init__(self, model_name='vinai/phobert-base-v2', type_backbone= 'bert', 
                 using_hidden_states= True, required_grad=False, dropout=0.1, 
                 hidden_dim=768, num_label=1, torch_dtype= torch.float16, device= None):

        self.model= CrossEncoder(model_name, type_backbone, using_hidden_states, 
                                 required_grad, dropout, hidden_dim, num_label)
        # self.model.to(device, dtype= torch_dtype)
        self.tokenizer= AutoTokenizer.from_pretrained(model_name, add_prefix_space= True, use_fast= True)
        self.device= device
        self.torch_dtype= torch_dtype

    def load_ckpt(self, path):
        self.model.load_state_dict(torch.load(path, map_location= 'cpu')['model_state_dict'])
        self.model.to(self.device, dtype= self.torch_dtype)

    def _preprocess(self):
        if self.model.training: 
            self.model.eval()
    
    def _preprocess_tokenize(self, text, max_length): 
        inputs= self.tokenizer.batch_encode_plus(text, return_tensors= 'pt', 
                            padding= 'longest', max_length= max_length, truncation= True)
        return inputs

    def predict(self, text: List[list[str]], max_length= 256):  # [[a, b], [c, d]]
        self._preprocess()
        batch_text= list(map(lambda x: self.tokenizer.sep_token.join([x[0], x[1]]), text))
        inputs= self._preprocess_tokenize(batch_text, max_length)

        with torch.no_grad(): 
            embedding= self.model(dict( (i, j.to(self.device)) for i,j in inputs.items()))
        
        return nn.Sigmoid()(torch.tensor(embedding))
        

    
    







        