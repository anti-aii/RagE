import torch
import numpy as np 

def _convert_data(data, return_tensors= 'pt'):
    """
    Convert input data to the desired tensor type.

    Parameters:
        data (torch.Tensor or numpy.ndarray): Input data to be converted.
        return_tensors (str, optional): Desired tensor type to return. Default is 'pt'.
            Possible values are 'pt' for PyTorch tensors and 'np' for numpy arrays.

    Returns:
        torch.Tensor or numpy.ndarray: Converted data in the specified tensor type.

    Raises:
        AssertionError: If `return_tensors` is not one of ['pt', 'np'].

    Example:
        >>> import numpy as np
        >>> import torch
        >>> data_np = np.array([1, 2, 3])
        >>> _convert_data(data_np, return_tensors='pt')
        tensor([1, 2, 3])
    """

    assert return_tensors in ['pt', 'np'] 
    if isinstance(data, torch.Tensor): 
        if return_tensors== 'pt': 
            return data 
        else:
            return data.cpu().numpy()

    if isinstance(data, np.ndarray): 
        if return_tensors== 'np': 
            return data 
        else: 
            return torch.tensor(data)
        
        

