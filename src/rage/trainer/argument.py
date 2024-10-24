from typing import Type, Union, Optional
import json 
import torch
from transformers import PreTrainedTokenizer

from ..utils.augment_text import TextAugment
from ..constant import CONFIG_DATASET, CONFIG_TRAIN

class ArgumentTrain: 
    def __init__(self, loss_function, gradient_accumlation_steps: int= 16, learning_rate: float= 1e-4, weight_decay: Optional[float]= 0.1, 
    eps: Optional[float]= 1e-6, warmup_steps: int= 150, epochs: Optional[int]= 1, optimizer= Type[torch.optim.Optimizer], 
    metrics: str= "loss", scheduler= None, data_parallel: bool= True
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
        self.data_parallel= data_parallel
    
    def save(self, file_name: str= CONFIG_TRAIN):
        with open(file_name, 'w', encoding= 'utf-8') as f: 
            json.dump(self.__dict__, f, ensure_ascii= False)


class ArgumentDataset: 
    def __init__(self, max_length: int= 256, batch_sampler: bool= False, drop_last: bool= False, advance_config_encode: Type[dict]= None, batch_size_per_gpu: int = 8, 
        shuffle: Optional[bool]= True, num_workers: int= 16, augment_data_function: TextAugment= None,
        pin_memory: Optional[bool]= True, prefetch_factor: int= 8, persistent_workers: Optional[bool]= True):
        self.max_length= max_length
        self.batch_sampler= batch_sampler 
        self.drop_last= drop_last
        self.advance_config_encode= advance_config_encode
        self.batch_size_per_gpu= batch_size_per_gpu 
        self.shuffle= shuffle
        self.num_workers= num_workers
        self.pin_memory= pin_memory
        self.prefetch_factor= prefetch_factor 
        self.persistent_workers= persistent_workers
        self.augment_data_function= augment_data_function
    
    def save(self, file_name: str= CONFIG_DATASET): 
        with open(file_name, 'w', encoding= 'utf-8') as f: 
            json.dump(self.__dict__, f, ensure_ascii= False)
            

class ArgumentMixTrainDataset: 
    def __init__(self, loss_function, gradient_accumlation_steps: int= 16, learning_rate: float= 1e-4, weight_decay: Optional[float]= 0.1, 
    eps: Optional[float]= 1e-6, warmup_steps: int= 150, epochs: Optional[int]= 1, optimizer= Type[torch.optim.Optimizer], 
    metrics: str= "loss", scheduler= None, data_parallel: bool= True,  max_length: int= 256, batch_sampler: bool= False, drop_last: bool= False,
    advance_config_encode: Type[dict]= None, batch_size_per_gpu: int = 8, shuffle: Optional[bool]= True, num_workers: int= 16, 
    augment_data_function: TextAugment= None, pin_memory: Optional[bool]= True, prefetch_factor: int= 8, persistent_workers: Optional[bool]= True
    ):  
        self.arg_train= ArgumentTrain(loss_function, gradient_accumlation_steps, learning_rate, weight_decay, eps, warmup_steps, epochs, 
                                      optimizer, metrics, scheduler, data_parallel)
        self.arg_dataset= ArgumentDataset(max_length, batch_sampler, drop_last, advance_config_encode, batch_size_per_gpu, shuffle, num_workers, 
                                          augment_data_function, pin_memory, prefetch_factor, persistent_workers)
        
    
    def save(self, file_name_train, file_name_dataset): 
        self.arg_train.save(file_name_train)
        self.arg_dataset.save(file_name_dataset)
        
    def get_att(self):
        return self.arg_train, self.arg_dataset