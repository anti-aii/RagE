import torch 
import loralib as lora


from typing import Union

def save_model(model: torch.nn.Module, filename: str, mode: str= "adapt_weight", 
               key= 'model_state_dict', metada: dict= None):
    assert mode in ['adapt_weight', 'full_weight']
    if  mode == 'adapt_weight':  # support LLMs
        weight= lora.lora_state_dict(model)
    elif mode== 'full_weight':  # only support bert based
        weight= model.state_dict()
    
    if metada: 
        torch.save({key: weight, 
                    'metadata': metada}, filename)
    elif key== None or key == "": 
        torch.save(weight, filename)

def load_model(model: torch.nn.Module, filename: str, key: str= 'model_state_dict'): 
    if key: 
        ckpt= torch.load(filename, map_location= "cpu")[key]
    else: 
        ckpt= torch.load(filename, map_location= "cpu")
    
    new_state_dict= {}
    for k, value in ckpt.items(): 
        if k.startswith('module.'): 
            new_state_dict[k[7:]]= value
        else:
            new_state_dict[k]= value
    model.load_state_dict(new_state_dict, strict= False)
        
