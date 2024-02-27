from typing import List
import torch 
import torch.nn as nn 
from transformers import AutoModel, AutoTokenizer
from ..componets import ExtraRoberta, load_backbone, PoolingStrategy

### Bi-encoder 
class BiEncoder(nn.Module):
    def __init__(self, model_name= 'vinai/phobert-base-v2', type_backbone= 'bert',
                 using_hidden_states= True, concat_embeddings= False, required_grad= True, 
                 strategy_pooling= "attention_context", dropout= 0.1, hidden_dim= 768, num_label= None):
        super(BiEncoder, self).__init__()

        self.using_hidden_states= using_hidden_states
        self.concat_embeddings= concat_embeddings
        self.strategy_pooling= strategy_pooling

        self.pooling= PoolingStrategy(strategy= strategy_pooling, units= hidden_dim)

        self.model= load_backbone(model_name, type_backbone= type_backbone, dropout= dropout,
                                using_hidden_states= using_hidden_states)

        if not required_grad:
            self.model.requires_grad_(False)
    
        # define 
        if self.using_hidden_states:
            self.extract= ExtraRoberta(method= 'mean')
        

        # dropout
        if strategy_pooling == "attention_context":
            self.drp1= nn.Dropout(p= dropout)
        
        # defind output 
        if self.concat_embeddings:  

            self.drp2= nn.Dropout(p= dropout)

            if not num_label: 
                self.fc= nn.Linear(hidden_dim * 2 + 1, 128) ## suggest using with cosine similarity loss based on sentence bert paper
            else:
                self.fc= nn.Linear(hidden_dim * 2 + 1, num_label)

            nn.init.xavier_uniform_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)

    def get_embedding(self, input): 
        embedding= self.model(**input)

        if self.using_hidden_states: 
            embedding= self.extract(embedding.hidden_states)
        else: 
            embedding= embedding.last_hidden_state

        # x= self.lnrom(embedding_enhance)
        if self.strategy_pooling == "attention_context": 
            embedding= self.drp1(embedding)

        x= self.pooling(embedding)

        return x 
    
    def forward(self, inputs, return_embeddings= False): 

        if return_embeddings:
            return [self.get_embedding(i) for i in inputs]
        
        if self.concat_embeddings and len(inputs) == 2: 
            x_left = self.get_embedding(inputs[0])
            x_right= self.get_embedding(inputs[1])
            x = torch.concat((x_left, x_right, torch.norm(x_right - x_left, p= 2, dim= -1).view(-1, 1)), dim= -1)
            x = self.drp2(x)
            x = self.fc(x)
            return x 
        
        raise ValueError("Input Error")

### Sentence Bert
### By default, while training sentence bert, we use cosine similarity loss 
class SentenceEmbedding: 
    def __init__(self, model_name= 'vinai/phobert-base-v2', type_backbone= 'bert',
                 using_hidden_states= True, concat_embeddings= False, required_grad= True, num_label= 1,
                 dropout= 0.1, hidden_dim= 768, torch_dtype= torch.float16, device= None):
    
        self.model= BiEncoder(model_name, type_backbone, using_hidden_states, concat_embeddings, 
                              required_grad, dropout, hidden_dim, num_label)
        # self.model.to(device, dtype= torch_dtype)
        self.tokenizer= AutoTokenizer.from_pretrained(model_name, use_fast= True, add_prefix_space= True)
        self.device= device 
        self.torch_dtype= torch_dtype
    
    def load_ckpt(self, path): 
        self.model.load_state_dict(torch.load(path, map_location= 'cpu')['model_state_dict'])
        self.model.to(self.device, dtype= self.torch_dtype)

    def _preprocess(self): 
        if self.model.training: 
            self.model.eval() 
    
    def _preprocess_tokenize(self, text, max_legnth= 256): 
        # 256 phobert, t5 512 
        inputs= self.tokenizer.batch_encode_plus(text, return_tensors= 'pt', 
                            padding= 'longest', max_length= max_legnth, truncation= True)
        return inputs
    
    def encode(self, text: List[str], max_length= 256): 
        # PhoBERT max length 256, T5 max length 512
        self._preprocess()
        # batch_text= list(map(lambda x: TextFormat.preprocess_text(x), text))
        inputs= self._preprocess_tokenize(text, max_length)
        # print(inputs)

        with torch.no_grad(): 
            embedding= self.model.get_embedding(dict( (i, j.to(self.device)) for i,j in inputs.items()))
                
        return torch.tensor(embedding) 




    


