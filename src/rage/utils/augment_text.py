import torch 
import numpy as np 
import logging 

class TextAugment: 
    """
    A class for text augmentation.

    This class serves as a base for text augmentation techniques.

    Args:
        None

    Methods:
        forward(text): Passes the input text through defined augmentation functions.

    Attributes:
        None
    """
    def forward(self, text):
       # pass text throught defined functions
       raise NotImplementedError

    def __call__(self, text): 
       # call forward function 
       return self.forward(text)
    

class NoiseMask(TextAugment):     
    """
    A class for applying noise masking augmentation to text.

    This class applies a noise masking technique to the input text.

    Args:
        p (float, optional): The probability of masking each word. Defaults to 0.1.
        mask_token (str, optional): The token used for masking. Defaults to "<mask>".

    Methods:
        forward(text): Applies noise masking augmentation to the input text.

    Attributes:
        p (float): The probability of masking each word.
        mask_token (str): The token used for masking.
    """

    def __init__(self, p= 0.1, mask_token= "<mask>"):
        self.p= p # using bernouli 
        self.mask_token= mask_token 
    
    def forward(self, text):
        """
        Applies noise masking augmentation to the input text.

        Args:
            text (str): The input text to be augmented.

        Returns:
            str: The augmented text with noise masking applied.
        """
        words = np.asarray(text.split())
        noise_mask = torch.bernoulli(torch.full((len(words),), self.p)).bool()  # Chuyển thành boolean
        words[noise_mask] = self.mask_token
        return " ".join(words)






