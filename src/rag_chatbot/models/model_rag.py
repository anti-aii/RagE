import torch 
import pandas as pd 
import datasets 

from abc import abstractmethod
from typing import Type, Union
from prettytable import PrettyTable 

from ..trainer.argument import ArgumentDataset, ArgumentTrain
from ..utils import (
    save_model, 
    load_model, 
    count_params_of_model
)
from ..utils.io_utils import print_out
from ..models import (
    BiEncoder, 
    CrossEncoder, 
    GenAnsModel
)
from ..trainer.trainer import (
    _TrainerBiEncoder,
    _TrainerCrossEncoder, 
    _TrainerGenAns
)


class ModelRag(torch.nn.Module): 
    ### currently support for BiEncoder, CrossEncoder
    @abstractmethod
    def _get_config_model_base(self):
        # return architecture base model, 
        pass 
    
    @abstractmethod
    def _get_config_addition_weight(self): 
        pass 
    
    @abstractmethod
    def _get_config(self):
        # return dict(self._get_config_model_base, **self._get_config_addition_weight)
        pass
    
    def _config2str(self): 
        config_name= str(self.__class__.__name__) + "Config("

        for key, value in self._get_config(): 
            config_name += f"{key}: {value}, "
        
        config_name= config_name.rstrip(', ')
        config_name += ")"

        return config_name 
    
    def save(self, path: str, mode: str= "auto_detect", limit_size= 6,
               size_limit_file= 3, storage_units= 'gb', key:str= 'model_state_dict', metada: dict= None):
        save_model(self, path, mode, limit_size, size_limit_file, storage_units, key,
                   metada)

    def load(self, path: str, multi_ckpt= False, key: str= 'model_state_dict'): 
        load_model(self, path, multi_ckpt, key)

    def summary_params(self): 
        count_params_of_model(self, count_trainable_params= False, 
                              return_result= False)

    def summary(self): 
        table= PrettyTable(['Layer (type)', 'Param'])
        for name, weight in self.named_children(): 
            table.add_row([f'{name} ({weight.__class__.__name__})', f'{count_params_of_model(weight):,}'])
        
        print_out(table)
    
    def compile(self, argument_train: Type[ArgumentTrain], argument_dataset: Type[ArgumentDataset]):
        if isinstance(self, BiEncoder): 
            self._trainer= _TrainerBiEncoder(self, argument_train, argument_dataset)
        if isinstance(self, CrossEncoder): 
            self._trainer= _TrainerCrossEncoder(self, argument_train, argument_dataset)
        if isinstance(self, GenAnsModel):
            self._trainer= _TrainerGenAns(self, argument_train, argument_dataset)
    
    @abstractmethod
    def forward(self): 
        pass 

    def fit(self, data_train: Union[str, pd.DataFrame, datasets.Dataset], data_eval: Union[str, pd.DataFrame, datasets.Dataset]= None,
        verbose= 1, use_wandb= True, step_save: int= 1000, path_save_ckpt_step: str= "step_ckpt.pt", path_save_ckpt_epoch: str= "best_ckpt.pt"):
        self._trainer.fit(data_train, data_eval, verbose, use_wandb, step_save, path_save_ckpt_step, path_save_ckpt_epoch)


    def eval(self, data):
        # currently not support 
        pass 
    
    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self):
        return self._config2str()
