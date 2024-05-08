import torch.nn as nn
from ..constant import EMBEDDING_IN_BATCH_NEGATIVES


class LossRAG(nn.Module): 
    def __init__(self): 
        super(LossRAG, self).__init__()

        self._pretty_name= None # default 
        self._task_name= EMBEDDING_IN_BATCH_NEGATIVES

    @property
    def pretty_name(self):
        return self._pretty_name 
    
    @pretty_name.setter 
    def pretty_name(self, value): 
        self._pretty_name= value 

    @property
    def task_name(self):
        return self._task_name

    @task_name.setter
    def task_name(self, value): 
        self._task_name= value

    def compile(self, model): 
        self.model= model 
    
    def _get_config_task(self):
        return {
            'task': self.task_name, 
            'pretty_name': self.pretty_name
        }
    
    def _get_config_params(self): 
        raise NotImplementedError
    
    def get_config(self): 
        raise {
            'loss_name': self.__class__.__name__, 
            'params': self._get_config_params(), 
        }