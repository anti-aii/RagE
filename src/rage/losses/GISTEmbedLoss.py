from typing import Iterable, Dict
import torch 
from torch import nn, Tensor
from .loss_rag import LossRAG
from ..constant import EMBEDDING_IN_BATCH_NEGATIVES


def cos_sim(a, b): 
    a_norm= torch.nn.functional.normalize(a, p= 2, dim= -1)
    b_norm= torch.nn.functional.normalize(b, p= 2, dim= -1)

    return torch.mm(a_norm, b_norm.transpose(0, 1))

class GITSEmbedLoss(LossRAG): 
    """ We innovate based on this paper: https://arxiv.org/abs/2402.16829
    """

    def __init__(
        self, 
        guide_model_function: None, 
        temp: int= 0.01  
    ): 

        self.guild_model_func= guide_model_function
        self.temp= temp 

        if not guide_model_function or not callable(guide_model_function): 
            raise ValueError(
                """You must define the function before passing it as a parameter to guide_model_func. 
                For example:
                class myfunc:
                    def __init__(self, model): 
                        self.model= model 
                    def __cal__(self, text): 
                        return model.encode(text)
                    
                loss= GITSEmbedLoss(guild_model_func= myfunc(), temp= 0.01)
                """
            )
        
        self.loss_fct= nn.CrossEntropyLoss()
        self.pretty_name= "gitsembed"
        self.task_name= EMBEDDING_IN_BATCH_NEGATIVES 
    
    def _get_config_params(self):
        return super()._get_config_params()

    
    def forward(self, features: Iterable[Dict[str, Tensor]]): 
        decode_text= [
            self.model.tokenizer.batch_decode(feature['input_ids']) for feature in features
        ]
        
        guide_embedding= [
            self.guild_model_func(feature) for feature in decode_text
        ] 
        embedding= self.model(features, return_embeddings= True)

        neg= None 
        if len(embedding) == 2:
            anchor, pos= embedding
            anchor_guide, pos_guide= guide_embedding
        else: 
            anchor, pos= embedding[:2]
            neg= torch.cat(embedding[2:])

            # guild molde 
            anchor_guide, pos_guide= guide_embedding[:2] 
            neg_guide= torch.cat(guide_embedding[2:])

        # model's similarities
        aa_score= cos_sim(anchor, anchor)
        ap_score= cos_sim(anchor, pos)
        # pp_score= cos_sim(pos, pos)

        ap_score_gold= ap_score.diagonal(dim1= 1, dim2= 2)

        # guide model's similarities
        aa_guide_score= cos_sim(anchor_guide, anchor_guide)
        ap_guide_socre= cos_sim(anchor_guide, pos_guide)
        # pp_guide_score= cos_sim(pos_guide, pos_guide) 

        ap_guide_gold_score= ap_guide_socre.diagonal(dim1= 1, dim2= 2).unsqueeze(-1)

        ap_score[ap_guide_socre >= ap_guide_gold_score]= -torch.inf
        aa_score[aa_guide_score >= ap_guide_gold_score]= -torch.inf
        # pp_score[pp_guide_score > ap_guide_gold_score]= -torch.inf

        ap_score.diagonal_scatter(ap_score_gold, dim1= 1, dim2= 2)

        score= [ap_score, aa_score]
        
        if neg: 
            an_score= cos_sim(anchor, neg)
            an_guide_score= cos_sim(anchor_guide, neg_guide)
            an_score[an_guide_score >= ap_guide_gold_score]= -torch.inf
            score.append(an_score)
    
        score= torch.cat(score, dim= 1)/ self.temp
        labels= torch.arange(len(score), device= score.device)

        return self.loss_fct(score, labels)






        

        