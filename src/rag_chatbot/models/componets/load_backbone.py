import torch 
from transformers import (
    AutoModelForTextEncoding,
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM
)

def load_backbone(model_name, type_backbone= 'mlm', using_hidden_states= True, dropout= 0.1,
                  torch_dtype= torch.float16, quantization_config= None): 

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

    return model 