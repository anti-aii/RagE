from typing import Union
from ..models import GenAnsModel, BiEncoder, CrossEncoder

def print_trainable_parameters(model: Union[GenAnsModel, BiEncoder, CrossEncoder]): 
    trainable_params = 0
    all_param = 0
    # if isinstance(model, CrossEncoder) or isinstance(model, BiEncoder): 
    #     params= model.parameters()
    # else: 
    #     params= model.model.parameters() 
    for _, param in model.parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )