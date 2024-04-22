import torch
import numpy as np 

def _convert_data(data, return_tensors= 'pt'):
    assert return_tensors in ['pt', 'np'] 
    if isinstance(data, torch.Tensor): 
        if return_tensors== 'pt': 
            return data 
        else:
            return data.numpy()

    if isinstance(data, np.ndarray): 
        if return_tensors== 'np': 
            return data 
        else: 
            return torch.tensor(data)
        
        

