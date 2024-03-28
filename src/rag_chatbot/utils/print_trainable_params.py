from typing import Type
import torch 

import sys
import logging 
logger= logging.Logger(__name__)

def print_trainable_parameters(model: Type[torch.nn.Module]): 
    logger.warning("This function will be removed in the future. Please use count_params_of_model")
    trainable_params = 0
    all_param = 0
    # if isinstance(model, CrossEncoder) or isinstance(model, BiEncoder): 
    #     params= model.parameters()
    # else: 
    #     params= model.model.parameters() 
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    sys.stdout.writelines(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    sys.stdout.flush()


def count_params_of_model(model: Type[torch.nn.Module], count_trainable_params= True, return_result= True): 
    all_param= 0
    trainable_params= 0

    for _, param in model.named_parameters(): 
        all_param += param.numel()
        if param.requires_grad and count_trainable_params:
            trainable_params += param.numel()

    if return_result: 
        return {
            'all_params': all_param, 
            'trainable_params': trainable_params
        }
    else: 
        sys.stdout.writelines(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )
        sys.stdout.flush()