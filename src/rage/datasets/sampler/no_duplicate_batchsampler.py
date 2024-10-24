import torch
import math
from torch.utils.data import Sampler

class NoDuplicatesBatchSampler(Sampler): 
    def __init__(
        self, 
        dataset, 
        batch_size, 
        shuffle: bool, 
        drop_last: bool, 
        *, 
        obverse: str= 'query'
    ): 
        self.dataset= dataset
        self.batch_size= batch_size
        self.shuffle= shuffle  
        self.obverse= obverse
        self.drop_last= drop_last
        
        assert self.obverse in ['query', 'query_positive'], "obvesre is only allowed to assign 'query' or 'query_positive' values"
            
    def shuffle_index(
        self, 
        data
    ): 
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        return torch.randperm(len(data), dtype= torch.int64, generator= generator)
    
    def __iter__(self):
        if self.shuffle:    
            index_dataset= self.shuffle_index(self.dataset)
        else: 
            index_dataset= torch.arange(len(self.dataset), dtype= torch.int64)
            
        batch_index= []
        duplicate_queue= []
        data_point= 0 
        text_in_batch= set()
        
        while len(index_dataset) > 0: 
            idx, index_dataset= int(index_dataset[0].item()), index_dataset[1:]
            if self.obverse=='query': 
                example= list(self.dataset.__getitem__(idx)[0]) # obverse on anchor or query
            elif self.obverse=='query_positive': 
                example= '<type>'.join(list(self.dataset.__getitem__(idx))[:1])
            
            if example.strip().lower() in text_in_batch:
                duplicate_queue.append(idx)
            else: 
                batch_index.append(idx)
                text_in_batch.add(example.strip().lower())

            if self.drop_last and (len(index_dataset) ==0 and len(batch_index) < self.batch_size):
                break 

            if len(batch_index)== self.batch_size: 
                yield batch_index
                batch_index= []
                text_in_batch= set()
                index_dataset= torch.cat((index_dataset, torch.tensor(duplicate_queue)))
                index_dataset= index_dataset[self.shuffle_index(index_dataset)]
            
            data_point+= 1
            if data_point >= self.batch_size: 
                data_point= 0
                index_dataset= index_dataset[self.shuffle_index(index_dataset)]

    def __len__(self): 
        return math.ceil(len(self.dataset)/self.batch_size)


