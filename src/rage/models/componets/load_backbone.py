from typing import Type

import torch 
from transformers import (
    AutoModelForTextEncoding,
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast
)

def load_backbone(model_name, type_backbone= 'mlm', using_hidden_states= True, dropout= 0.1,
                  torch_dtype= torch.float16, quantization_config= None, load_tokenizer: bool= True): 

    assert type_backbone in ['mlm', 'casual_lm', 'seq2seq']

    if type_backbone== 'mlm': 
        model= AutoModelForTextEncoding.from_pretrained(model_name, output_hidden_states= using_hidden_states,
                        hidden_dropout_prob = dropout, attention_probs_dropout_prob = dropout, quantization_config= quantization_config, 
                        torch_dtype= torch_dtype, device_map= 'auto' if quantization_config else 'cpu')
    
    if type_backbone == 'casual_lm': 
        model= AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype= torch_dtype, device_map= 'auto' if quantization_config else 'cpu', 
            quantization_config= quantization_config)

    if type_backbone== "seq2seq": 
        model= AutoModelForSeq2SeqLM.from_pretrained(
            model_name, torch_dtype= torch_dtype, device_map= 'auto' if quantization_config else 'cpu', 
            quantization_config= quantization_config)

    ## tokenizer , as default the tokenizer of model embedding or bert have config: self.tokenizer= AutoTokenizer.from_pretrained(model_name, add_prefix_space= True, use_fast= True)
    if load_tokenizer:
        if type_backbone== "mlm":
            tokenizer= AutoTokenizer.from_pretrained(model_name, add_prefix_space= True, use_fast= True)
        else: 
            tokenizer= AutoTokenizer.from_pretrained(model_name, use_fast= True)
    else: 
        tokenizer= None
        
    return model, tokenizer


def selective_model_base(
    model_base: Type[torch.nn.Module], 
    tokenizer_base: Type[AutoTokenizer], 
    **kwargs
):
    if isinstance(model_base, torch.nn.Module):
        model= model_base
    load_tokenizer= True
    
    if isinstance(tokenizer_base, PreTrainedTokenizer) or isinstance(tokenizer_base, PreTrainedTokenizerFast): 
        tokenizer= tokenizer_base
        load_tokenizer= False
    
    
    if isinstance(kwargs['model_name'], str) and model_base == None: 
        model, tokenizer_rp= load_backbone(
            **kwargs, 
            load_tokenizer= load_tokenizer)
    
    if load_tokenizer: 
        return model, tokenizer_rp
    else: 
        return model, tokenizer