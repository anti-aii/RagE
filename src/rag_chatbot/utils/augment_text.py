import torch 
import numpy as np 
import logging 

class TextAugment: 
    def forward(self, text):
       # pass text throught defined functions
       pass 

    def __call__(self, text): 
       # call forward function 
       return self.forward(text)
    

class NoiseMask(TextAugment): 
    def __init__(self, p= 0.1, mask_token= "<mask>"):
        self.p= p # using bernouli 
        self.mask_token= mask_token 
    
    def forward(self, text):
        words = np.asarray(text.split())
        noise_mask = torch.bernoulli(torch.full((len(words),), self.p)).bool()  # Chuyển thành boolean
        words[noise_mask] = self.mask_token
        return " ".join(words)






