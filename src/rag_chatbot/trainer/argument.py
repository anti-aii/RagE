from typing import Type, Union, Optional
import pandas as pd 
import torch 
from transformers import PreTrainedTokenizer
from datasets import Dataset

from ..utils.augment_text import TextAugment


class ArgumentTrain: 
    def __init__(self, loss_function, gradient_accumlation_steps: int= 16, learning_rate: float= 1e-4, weight_decay: Optional[float]= 0.1, 
    eps: Optional[float]= 1e-6, warmup_steps: int= 150, epochs: Optional[int]= 1, optimizer= Type[torch.optim.Optimizer], 
    metrics: str= "loss", scheduler= None, torch_compile= False, backend_torch_compile: str= None, data_parallel: bool= True
    ):  
        self.loss_function= loss_function
        self.grad_accum= gradient_accumlation_steps 
        self.optimizer= optimizer 
        self.scheduler= scheduler
        self.lr= learning_rate
        self.eps= eps
        self.weight_decay= weight_decay
        self.warmup_steps= warmup_steps
        self.metrics= metrics
        self.epochs= epochs 
        self.torch_compile= torch_compile
        self.backend_torch_compile= backend_torch_compile 
        self.data_parallel= data_parallel


class ArgumentDataset: 
    def __init__(self, tokenizer: Union[str, PreTrainedTokenizer], max_length: int= 256, batch_size_per_gpu: int = 8, 
        shuffle: Optional[bool]= True, num_workers: int= 16, augment_data_function: TextAugment= None,
        pin_memory: Optional[bool]= True, prefetch_factor: int= 8, persistent_workers: Optional[bool]= True):
        self.max_length= max_length
        self.tokenizer= tokenizer
        self.batch_size_per_gpu= batch_size_per_gpu 
        self.shuffle= shuffle
        self.num_workers= num_workers
        self.pin_memory= pin_memory
        self.prefetch_factor= prefetch_factor 
        self.persistent_workers= persistent_workers
        self.augment_data_function= augment_data_function
        