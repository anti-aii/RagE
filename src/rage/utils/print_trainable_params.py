from typing import Type
import torch
import logging  
from .io_utils import _print_out


logger= logging.Logger(__name__)

def _print_trainable_parameters(model: Type[torch.nn.Module]): 
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

    _print_out(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def _count_params_of_model(sequential, count_trainable_params= True, return_result= True):
    """
    Count the total number of parameters in a PyTorch model.

    Parameters:
    - sequential (torch.nn.Sequential): The sequential model.
    - count_trainable_params (bool, optional): If True, count only trainable parameters. Default is True.
    - return_result (bool, optional): If True, return the result as a dictionary containing all parameters and trainable parameters.
                                      If False, print the result. Default is True.

    Returns:
    If return_result is True:
        dict: A dictionary containing the count of all parameters and trainable parameters.
            {
                'all_params': int,          # Total number of parameters in the model
                'trainable_params': int     # Total number of trainable parameters in the model
            }
    If return_result is False: (prints the result instead of returning)
        None

    Note:
    - The count includes both trainable and non-trainable parameters.
    - If return_result is False, the function prints the count of trainable parameters, total parameters, and the percentage of trainable parameters.
    """ 
    all_param= 0
    trainable_params= 0

    for _, param in sequential.named_parameters(): 
        all_param += param.numel()
        if param.requires_grad and count_trainable_params:
            trainable_params += param.numel()

    if return_result: 
        return {
            'all_params': all_param, 
            'trainable_params': trainable_params
        }
    else: 
        _print_out(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )