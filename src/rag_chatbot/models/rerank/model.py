from typing import List
import torch 
import torch.nn as nn 
from transformers import AutoTokenizer
from ..componets import AttentionWithContext, ExtraRoberta, load_backbone


### Cross-encoder
class CrossEncoder(nn.Module): 
    # using 
    def __init__(self, model_name= 'vinai/phobert-base-v2', type_backbone= 'bert',
                 using_hidden_states= True, required_grad= False, 
                 dropout= 0.1, hidden_dim= 768, num_label= 1):
        super(CrossEncoder, self).__init__()
        
        self.using_hidden_states= using_hidden_states
        self.model= load_backbone(model_name, type_backbone= type_backbone, 
                                using_hidden_states= using_hidden_states)

        if not required_grad:
            self.model.requires_grad_(False)
    
        # define 
        if self.using_hidden_states:
            self.extract= ExtraRoberta(method= 'mean')
        self.attention_context= AttentionWithContext(units= hidden_dim)

        # dropout 
        self.drp1= nn.Dropout(p= dropout)
        self.drp2= nn.Dropout(p= dropout)

        # defind output 
        self.fc= nn.Linear(hidden_dim, num_label)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def get_embedding(self, inputs):
        embedding= self.model(**inputs)

        if self.using_hidden_states: 
            embedding= self.extract(embedding.hidden_states)

        x= self.drp1(embedding)
        x= self.attention_context(x)

        return x 
    
    
    def forward(self, inputs): 
        x= self.get_embedding(inputs)
        x= self.drp2(x)
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
        

    
    







        