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
        generator = torch.Generator().manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        return torch.tensor(data)[torch.randperm(len(data), generator=generator)].tolist()

    
    def __iter__(self):    
        index_dataset= torch.arange(len(self.dataset), dtype= torch.int64).tolist()
        if self.shuffle: 
            self.shuffle_index(index_dataset)
            
        batch_index, text_in_batch= [], set()
    
        while index_dataset: 
            idx= index_dataset.pop(0)
            example = list(self.dataset[idx])[0].strip().lower() if self.obverse == 'query' else '<type>'.join(list(self.dataset[idx])[:1]).strip().lower()
            
            if example in text_in_batch:
                continue

            batch_index.append(idx)
            text_in_batch.add(example)

            if len(batch_index)== self.batch_size: 
                yield batch_index
                batch_index, text_in_batch= [], set() 
                self.shuffle_index(index_dataset)
        
        if not self.drop_last and batch_index: 
            yield batch_index
            

    def __len__(self): 
        return math.ceil(len(self.dataset)/self.batch_size)


