import os 
import json 
import torch 

from collections import OrderedDict
from typing import Type
import logging 

from .io_utils import _ensure_dir, _count_capacity_bit_weight
from .print_trainable_params import _count_params_of_model



def _save_only_trainable_weight(model: Type[torch.nn.Module], filename: str):
    params= OrderedDict()

    for name, weight in model.named_parameters(): 
        if weight.requires_grad: 
            params[name]= weight
    
    if len(params) == 0: 
        raise ValueError("No weight trainable have been saved")
        return None 

    return params


def _save_split_weight(model: Type[torch.nn.Module], path:str, size_limit_file= 3, storage_units= 'gb'): 
    current_weights= OrderedDict()
    current_count_params= 0
    number_ckpt= 0 
    model_index=[]

    _ensure_dir(path, create_path= True)
    path= os.path.join(os.getcwd(), path)

    for name, weight in model.named_parameters(): 
        current_count_params += weight.numel()
        current_weights[name]= weight

        if (_count_capacity_bit_weight(current_count_params, 
            storage_units= storage_units) < size_limit_file) or (len(current_weights) != 0): 
            current_count_params =0 
            current_weights= OrderedDict()
            
            model_name= "model_{:04d}.bin".format(number_ckpt)
            model_index.append(model_name)
            torch.save(current_weights, os.path.join(path, model_name))
            number_ckpt += 1

    ### saving model index 
    with open(os.path.join(path, "model_index.json"), 'w') as f: 
        json.dump(model_index, f)

def load_split_weight(path: str): 
    if not _ensure_dir(path, create_path= False): 
        raise ValueError("Not exits checkpoint path")
    
    params= OrderedDict()
    path= os.path.join(os.getcwd(), path)

    with open(os.path.join(path, "model_index.json")) as f: 
        model_index= json.load(f)
    
    for file in model_index: 
        params.update(torch.load(os.path.join(path, file), map_location= 'cpu'))

    
    return params
    

def save_model(model: Type[torch.nn.Module], path: str, mode: str= "auto_detect", limit_size= 6,
               size_limit_file= 3, storage_units= 'gb', key:str= 'model_state_dict', metada: dict= None):
    assert mode in ['trainable_weight', 'full_weight', 'multi_ckpt', 'auto_detect']

    # _ensure_dir(path, create_path= True)
    mode1, mode2= None, None

    if mode== 'auto_detect': 
        number_params= _count_params_of_model(model, return_result= True)
        if (number_params['trainable_params'] / number_params['all_params']) < 0.5: 
            mode1= 'trainable_weight'

        if _count_capacity_bit_weight(number_params['trainable_params'], storage_units= storage_units) > limit_size:
            mode2= 'multi_ckpt'

        if mode2: 
            if mode1:
                mode= 'trainable_weight'
            else:
                mode= 'multi_ckpt'
        else: 
            if mode1: 
                mode= 'trainable_weight'
            else: 
                mode= 'full_weight'
        

    if  mode == 'trainable_weight':  # support LLMs
        weight= _save_only_trainable_weight(model, filename= path)
    elif mode== 'full_weight':  # only support bert based
        weight= model.state_dict()
    elif mode== 'multi_ckpt':
        _save_split_weight(model, path= path, size_limit_file= size_limit_file,
                          storage_units= storage_units)
        return None
        
    if key== None or key == "": # not support save metadata 
        torch.save(weight, path)
    else: 
        torch.save({key: weight,
                    'metadata': metada}, path)
    

def load_model(model: Type[torch.nn.Module], path: str, multi_ckpt= False, key: str= 'model_state_dict'): 
    if multi_ckpt:
        model.load_state_dict(load_split_weight(path= path))
        return ModuleNotFoundError

    if key: 
        ckpt= torch.load(path, map_location= "cpu")[key]
    else: 
        ckpt= torch.load(path, map_location= "cpu")
    
    new_state_dict= {}
    for k, value in ckpt.items(): 
        if k.startswith('module.'): 
            new_state_dict[k[7:]]= value
        else:
            new_state_dict[k]= value
    model.load_state_dict(new_state_dict, strict= False)
        
