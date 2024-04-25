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
    

def save_model(model: Type[torch.nn.Module], path: str, mode: str= "trainable_weight", limit_size= 6,
               size_limit_file= 3, storage_units= 'gb', key:str= 'model_state_dict', metadata: dict= None):
    
    """
    Save the model parameters or weights to a file.

    Args:
        model (Type[torch.nn.Module]): The PyTorch model to save.
        path (str): The file path where the model will be saved.
        mode (str, optional): The mode for saving the model. Defaults to "trainable_weight".
            Options are: "trainable_weight", "full_weight", "multi_ckpt", "auto_detect".
        limit_size (int, optional): The limit size for saving weights in gigabytes (GB). Defaults to 6.
        size_limit_file (int, optional): The size limit for each split file when saving weights in multi-ckpt mode. Defaults to 3.
        storage_units (str, optional): The storage units to use for size limits. Defaults to 'gb'.
        key (str, optional): The key to use when saving the model state dictionary. Defaults to 'model_state_dict'.
        metadata (dict, optional): Additional metadata to save along with the model. Defaults to None.

    Raises:
        AssertionError: If mode is not one of ['trainable_weight', 'full_weight', 'multi_ckpt', 'auto_detect'].

    Returns:
        None: If mode is 'multi_ckpt', otherwise returns None.
    """
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
                    'metadata': metadata}, path)
    

def load_model(model: Type[torch.nn.Module], path: str, multi_ckpt= False, key: str= 'model_state_dict',):
    """
    Load weights of a PyTorch model from a specified path.

    Args:
        model (Type[torch.nn.Module]): The PyTorch model to load weights into.
        path (str): The path to the checkpoint file.
        multi_ckpt (bool, optional): Flag indicating whether the checkpoint file contains multiple models. 
            Defaults to False.
        key (str, optional): The key in the checkpoint file under which the model's state dictionary is stored.
            Defaults to 'model_state_dict'.

    Returns:
        None: The function does not return anything, it directly loads the model weights.
    """ 
    if multi_ckpt:
        model.load_state_dict(load_split_weight(path= path))

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
        
