# I rewirte RandomSampler 
import torch 
import math 
from torch.utils.data import Sampler

class RandomBatchSampler(Sampler): 
    def __init__(
        self,
        dataset, 
        batch_size,  
        drop_last: bool
    ):
        self.dataset= dataset
        self.batch_size= batch_size
        self.drop_last= drop_last
        
    def shuffle_index(
        self, 
        data
    ): 
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        return torch.randperm(len(data), dtype= torch.int64, generator= generator)
    
    def __iter__(self): 
        index_dataset= self.shuffle_index(self.dataset)
        batch_index= []
        
        while len(index_dataset) > 0:
            idx, index_dataset= index_dataset[0], index_dataset[1:]
            batch_index.append(idx)
            
            if self.drop_last and (len(index_dataset) ==0 and len(batch_index) < self.batch_size):
                break 
            
            if len(batch_index)== self.batch_size: 
                yield batch_index 
                batch_index= []
            
    def __len__(self):
        return math.ceil(len(self.dataset)/self.batch_size)