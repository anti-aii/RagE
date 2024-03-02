from typing import Type, Optional, Union
import pandas as pd 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler 
from transformers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm 
from bitsandbytes.optim import PagedAdamW8bit
from datasets import Dataset
import loralib as lora 
import wandb 

from ..models import (
    CrossEncoder, 
    BiEncoder, 
    GenAnsModel
)
from ..datasets import (
    GenAnsCollate, 
    GenAnsDL, 
    SentABCollate, 
    SentABDL
)
from ..losses import (
    CosineSimilarityLoss, 
    MSELogLoss, 
    ContrastiveLoss,
    TripletLoss, 
    InBatchNegativeLoss
)

from ..constant import (
    EMBEDDING_RANKER_NUMERICAL,
    EMBEDDING_CONTRASTIVE,
    EMBEDDING_TRIPLET, 
    EMBEDDING_IN_BATCH_NEGATIVES
)

from ..losses import (
    BINARY_CROSS_ENTROPY_LOSS,
    CATEGORICAL_CROSS_ENTROPY_LOSS,
    MSE_LOGARIT,
    COSINE_SIMILARITY_LOSS, 
    CONTRASTIVE_LOSS,
    TRIPLET_LOSS, 
    IN_BATCH_NEGATIVES_LOSS
)
from ..constant import RULE_LOSS_TASK
from ..utils.augment_text import TextAugment
from ..utils import save_model

import logging 
logging.basicConfig(level= logging.INFO)


## default eval is loss 

class ArgumentTrainer: 
    pass 

class ArgumentDataset: 
    pass 

class Trainer: 
    def __init__(self, 
        model, tokenizer_name: Type[str], data_train: Union[pd.DataFrame, str, Dataset], device: Type[torch.device], 
        data_eval: Union[pd.DataFrame, str, Dataset]= None, batch_size: int = 8, shuffle: Optional[bool]= True, num_workers: int= 16, 
        pin_memory: Optional[bool]= True, prefetch_factor: int= 8, persistent_workers: Optional[bool]= True, 
        gradient_accumlation_steps: int= 16, learning_rate: float= 1e-4, weight_decay: Optional[float]= 0.1, 
        eps: Optional[float]= 1e-6, warmup_steps: int= 150, epochs: Optional[int]= 1, path_ckpt_step: Optional[str]= 'checkpoint.pt',
        use_wandb: bool= True  
        ):

        # base 
        self.model_lm= model 

        # dataset
        self.data_train= data_train
        self.data_eval= data_eval

        self.dataloader_train= None
        self.dataloader_eval= None 
        # train args
        self.batch_size= batch_size
        self.shuffle= shuffle
        self.num_workers= num_workers
        self.pin_memory= pin_memory
        self.prefetch_factor= prefetch_factor
        self.persistent_workers= persistent_workers 
        self.grad_accum= gradient_accumlation_steps 
        self.lr= learning_rate 
        self.weight_decay= weight_decay 
        self.eps= eps 
        self.warmup_steps= warmup_steps
        self.epochs= epochs
        
        # device
        self.device= device 

        # optim 
        self.scheduler= None 
        self.total_steps= None 
        self.optimizer= None

        # mixer precision 
        self.scaler= None 

        # path ckpt 
        self.ckpt_step= path_ckpt_step

        # other 
        self.use_wandb= use_wandb 

    def _setup_dataset(self): 
        pass

    def _setup_dataloader(self): 
        train_dataset, eval_dataset= self._setup_dataset()
        self.dataloader_train= DataLoader(train_dataset, batch_size= self.batch_size,
                                          collate_fn= self.collate, shuffle= self.shuffle,
                                          num_workers= self.num_workers, pin_memory= self.pin_memory, 
                                          prefetch_factor= self.prefetch_factor, persistent_workers= self.persistent_workers)
        
        if self.data_eval: 
            self.dataloader_eval= DataLoader(eval_dataset, batch_size= self.batch_size,
                                             collate_fn= self.collate, shuffle= False)
    
    def _setup_optim(self): 
        self.optimizer= PagedAdamW8bit(self.model_lm.parameters(), self.lr, weight_decay= self.weight_decay, 
                                      eps= self.eps)
        step_epoch = len(self.dataloader_train)
        self.total_steps= int(step_epoch / self.grad_accum) * self.epochs

        self.scheduler= get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps= self.warmup_steps,
                                                        num_training_steps= self.total_steps)
        
    def _setup_mxprecision(self): 
        self.scaler= GradScaler()
    
    def _setup(self): 
        self._setup_dataloader()
        self._setup_optim() 
        self._setup_mxprecision()
    
    def _compute_loss(self, data):
        pass 

    def _save_ckpt(self, path, metadata): 
        if isinstance(self.model_lm, GenAnsModel):
            # LLMs
            save_model(self.model_lm.model, filename= path, mode= "adapt_weight", 
                       key= "lora", metada= metadata)
        else: 
            save_model(self.model_lm, filename= path, mode= "full_weight", 
                       key= "model_state_dict", metada= metadata)

    def _train_on_epoch(self, index_grad, verbose: Optional[int]= 1, step_save: Optional[int]= 1000): 
        self.model_lm.train()
        total_loss, total_count= 0, 0
        step_loss, step_fr= 0, 0

        for idx, data in enumerate(self.dataloader_train): 
            with autocast():
                loss= self._compute_loss(data)
                loss /= self.grad_accum
            self.scaler.scale(loss).backward()

            step_loss += loss.item() * self.grad_accum
            step_fr +=1 

            if ((idx + 1) % self.grad_accum == 0) or (idx + 1 ==len(self.dataloader_train)): 
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model_lm.parameters(), 1.) 
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none= True)
                self.scheduler.step() 

                if self.use_wandb:
                    wandb.log({"Train loss": step_loss / step_fr})
            
            total_loss += loss.item() 
            total_count += 1 

            if (idx + 1) % (self.grad_accum * verbose) == 0: 
                print(f'Step: [{index_grad[0]}/{self.total_steps}], Loss: {step_loss / step_fr}')
                step_loss = 0 
                step_fr = 0 
                index_grad[0] += 1 

            if (idx + 1) % step_save ==0:
                self._save_ckpt(path= self.ckpt_step, metadata= {
                    'step': idx + 1, 
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),   
                })
        
        return (total_loss / total_count) * self.grad_accum
    
    def _evaluate(self):
        pass 

    def fit(self, verbose: Type[int]= 1, step_save: Type[int]= 1000, path_ckpt_epoch: Type[str]= 'best_ckpt.pt'):
        index_grad= [1] 
        print('=' * 10 + ' Setup training' + '=' * 10)
        self._setup()
        print('=' * 10 + ' Begin training ' + '=' * 10)
        log_loss = 1e9
        for epoch in range(1, self.epochs + 1): 
            train_loss= self._train_on_epoch(index_grad, verbose, step_save)
            # val_loss = evaluate()
            print('-' * 59)
            print('-' * 59 + f'End of epoch {epoch} - loss: {train_loss}' + '-' * 59)
            print('-' * 59)
            
            if self.path_dataeval:
                print('=' * 10 + ' EVALUATE ' + '=' * 10)
                val_loss= self._evaluate()
                print(f'Evaluate loss: {val_loss}')

                if val_loss < log_loss: 
                    log_loss = val_loss
                    print(f'Saving checkpoint have best {log_loss}')
                    self._save_ckpt(path= path_ckpt_epoch)
                        


            if train_loss < log_loss: # saving 
                log_loss = train_loss
                print(f'Saving checkpoint have best {log_loss}')
                self._save_ckpt(path= path_ckpt_epoch)
        

class TrainerGenAns(Trainer):
    def __init__(self, model: Type[GenAnsModel], tokenizer_name: type[str], data_train: Union[pd.DataFrame, str], 
                 device: type[torch.device], max_length: int, data_eval: Union[pd.DataFrame, str] = None, batch_size: int = 8, shuffle: bool | None = True, 
                 num_workers: int = 16, pin_memory: bool | None = True, prefetch_factor: int = 8, persistent_workers: bool | None = True, 
                 gradient_accumlation_steps: int = 16, learning_rate: float = 0.0001, weight_decay: float | None = 0.1, 
                 eps: float | None = 0.000001, warmup_steps: int = 150, epochs: int | None = 1, path_ckpt_step: str | None = 'checkpoint.pt', 
                 use_wandb: bool = True):
        
        super().__init__(model, tokenizer_name, data_train, device, data_eval, batch_size, shuffle, num_workers, 
                         pin_memory, prefetch_factor, persistent_workers, gradient_accumlation_steps,
                         learning_rate, weight_decay, eps, warmup_steps, epochs, path_ckpt_step, use_wandb)
        
        self.collate= GenAnsCollate(tokenizer_name, max_length)

    def _setup_dataset(self): 
        train_dataset= GenAnsDL(self.data_train)

        if self.data_eval: 
            return train_dataset, GenAnsDL(self.data_eval)
        
        return train_dataset, None 
    
    def _compute_loss(self, data):
        loss= self.model_lm(input_ids= data['x_ids'].to(self.device, non_blocking=True), 
                          attention_mask= data['x_mask'].to(self.device, non_blocking=True), 
                          labels= data['label'].to(self.device, non_blocking=True)).loss
        
        return loss 
        
    def _evaluate(self): 
        self.model_lm.eval()
        total_loss, total_count= 0, 0
        with torch.no_grad(): 
            for idx, data in enumerate(self.dataloader_eval): 
                loss= self._compute_loss(data)
                total_loss += loss.item() 
                if self.use_wandb: 
                    wandb.log({"Eval loss": loss.item()})
                total_count += 1 
        
        return total_loss / total_count 
        

class TrainerBiEncoder(Trainer):  ## support 
    def __init__(self, model: Type[BiEncoder], tokenizer_name: type[str], data_train: Union[pd.DataFrame, str], device: type[torch.device], 
                max_length: Type[int]= 256, type_backbone: str= 'bert', augment_func: Type[TextAugment]= None, 
                data_eval: Union[pd.DataFrame, str]= None, batch_size: int = 8, shuffle: bool | None = True, 
                num_workers: int = 16, pin_memory: bool | None = True, prefetch_factor: int = 8, persistent_workers: bool | None = True, 
                gradient_accumlation_steps: int = 16, learning_rate: float = 0.0001, weight_decay: float | None = 0.1, 
                eps: float | None = 0.000001, warmup_steps: int = 150, epochs: int | None = 1, path_ckpt_step: str | None = 'checkpoint.pt', loss: str = 'cosine_embedding', 
                use_wandb: bool = True):
        
        super().__init__(model, tokenizer_name, data_train, device, data_eval, batch_size, shuffle, num_workers, 
                        pin_memory, prefetch_factor, persistent_workers, gradient_accumlation_steps,
                        learning_rate, weight_decay, eps, warmup_steps, epochs, path_ckpt_step, use_wandb)
        
        
        self.loss= loss 
        self.task= RULE_LOSS_TASK[self.loss]
        
        ### variant loss support for training embedding model 
        assert loss in [BINARY_CROSS_ENTROPY_LOSS, 
                        CATEGORICAL_CROSS_ENTROPY_LOSS, 
                        COSINE_SIMILARITY_LOSS, 
                        CONTRASTIVE_LOSS,
                        TRIPLET_LOSS,
                        IN_BATCH_NEGATIVES_LOSS]
        
        self.criterion= {
            BINARY_CROSS_ENTROPY_LOSS: nn.BCEWithLogitsLoss(),
            CATEGORICAL_CROSS_ENTROPY_LOSS: nn.CrossEntropyLoss(),
            COSINE_SIMILARITY_LOSS: CosineSimilarityLoss(), 
            CONTRASTIVE_LOSS: ContrastiveLoss(margin= 0.5), 
            TRIPLET_LOSS: TripletLoss(margin= 0.5), 
            IN_BATCH_NEGATIVES_LOSS: InBatchNegativeLoss(temp= 0.05)

        }[loss]
        
        self.collate= SentABCollate(tokenizer_name, mode= "bi_encoder", type_backbone= type_backbone, 
                                    max_length= max_length, task= self.task, augment_func= augment_func)
    
    def _setup_dataset(self):
        train_dataset= SentABDL(self.data_train, task= self.task)

        if self.data_eval: 
            return train_dataset, SentABDL(self.data_eval, task= self.task)
        
        return train_dataset, None

    def _compute_loss(self, data):        
        if self.task != EMBEDDING_RANKER_NUMERICAL: 

            if self.loss == CONTRASTIVE_LOSS: 
                output= self.model_lm(
                    (
                    dict((i, j.to(self.device, non_blocking=True)) for i, j in data['sent1'].items() if i in ['input_ids', 'attention_mask']),
                    dict((i, j.to(self.device, non_blocking=True)) for i, j in data['sent2'].items() if i in ['input_ids', 'attention_mask'])
                    ),
                    return_embeddings= True
                )
                return self.criterion(output[0], output[1], data['label'])
            

            if self.loss == TRIPLET_LOSS: 
                output= self.model_lm(
                    (
                    dict((i, j.to(self.device, non_blocking=True)) for i, j in data['anchor'].items() if i in ['input_ids', 'attention_mask']),
                    dict((i, j.to(self.device, non_blocking=True)) for i, j in data['pos'].items() if i in ['input_ids', 'attention_mask']),
                    dict((i, j.to(self.device, non_blocking=True)) for i, j in data['neg'].items() if i in ['input_ids', 'attention_mask'])
                    ),
                    return_embeddings= True
                )
                return self.criterion(*output)
            
            if self.loss == IN_BATCH_NEGATIVES_LOSS: 
                data_input= [
                    dict((i, j.to(self.device, non_blocking=True)) for i, j in data['anchor'].items() if i in ['input_ids', 'attention_mask']),
                    dict((i, j.to(self.device, non_blocking=True)) for i, j in data['pos'].items() if i in ['input_ids', 'attention_mask']),
                ]

                if len(data) > 2: 
                    for i in range(len(data)-2): 
                        data_input.append(
                            dict((i, j.to(self.device, non_blocking=True)) for i, j in data[f'hard_neg_{i+1}'].items() if i in ['input_ids', 'attention_mask'])
                        )
                
                output= self.model_lm(data_input, return_embeddings= True)
                return self.criterion(output)
            
        elif self.task == EMBEDDING_RANKER_NUMERICAL: 
            label= data['label'].to(self.device, non_blocking=True)
            data_input= (
                dict((i, j.to(self.device, non_blocking=True)) for i, j in data['x_1'].items() if i in ['input_ids', 'attention_mask']),
                dict((i, j.to(self.device, non_blocking=True)) for i, j in data['x_2'].items() if i in ['input_ids', 'attention_mask']),
            )

            if self.loss== COSINE_SIMILARITY_LOSS: 
                output= self.model_lm(data_input, return_embeddings= True)
                return self.criterion(output[0], output[1], label.to(dtype= torch.float32))
            
            output= self.model_lm(data_input)

            if self.loss== BINARY_CROSS_ENTROPY_LOSS:
                output= output.view(-1,)
                label= label.to(dtype= torch.float32)
            
            return self.criterion(output, label) 
            
    
    def _evaluate(self): 
        self.model_lm.eval()
        total_loss, total_count= 0, 0
        with torch.no_grad(): 
            for idx, data in enumerate(self.dataloader_eval): 
                loss= self._compute_loss(data)
                total_loss += loss.item() 
                if self.use_wandb: 
                    wandb.log({"Eval loss": loss.item()})
                total_count += 1 
        
        return total_loss / total_count 
    

class TrainerCrossEncoder(Trainer):
    def __init__(self, model:Type[CrossEncoder], tokenizer_name: type[str], data_train: Union[pd.DataFrame, str], device: type[torch.device], 
                max_length: Type[int]= 256, type_backbone: str= 'bert', augment_func: Type[TextAugment]= None,
                data_eval: Union[pd.DataFrame, str]= None, batch_size: int = 8, shuffle: bool | None = True, 
                num_workers: int = 16, pin_memory: bool | None = True, prefetch_factor: int = 8, persistent_workers: bool | None = True, 
                gradient_accumlation_steps: int = 16, learning_rate: float = 0.0001, weight_decay: float | None = 0.1, 
                eps: float | None = 0.000001, warmup_steps: int = 150, epochs: int | None = 1, path_ckpt_step: str | None = 'checkpoint.pt', loss: str= 'sigmoid_crossentropy',
                use_wandb: bool = True):
        
        super().__init__(model, tokenizer_name, data_train, device, data_eval, batch_size, shuffle, num_workers, 
                        pin_memory, prefetch_factor, persistent_workers, gradient_accumlation_steps,
                        learning_rate, weight_decay, eps, warmup_steps, epochs, path_ckpt_step, use_wandb)
        
        self.loss= loss 
        self.task= RULE_LOSS_TASK[self.loss]
        
        assert loss in [BINARY_CROSS_ENTROPY_LOSS, 
                        CATEGORICAL_CROSS_ENTROPY_LOSS, 
                        MSE_LOGARIT]
        
        self.criterion= {
            BINARY_CROSS_ENTROPY_LOSS: nn.BCEWithLogitsLoss(),
            CATEGORICAL_CROSS_ENTROPY_LOSS: nn.CrossEntropyLoss(),
            MSE_LOGARIT: MSELogLoss()
        }[loss]

        self.collate= SentABCollate(tokenizer_name, mode= "cross_encoder", type_backbone= type_backbone, 
                                    max_length= max_length, task= self.task, augment_func= augment_func)
        
    def _setup_dataset(self):
        train_dataset= SentABDL(self.data_train, task= self.task)

        if self.data_eval: 
            return train_dataset, SentABDL(self.data_eval, task= self.task)
        
        return train_dataset, None

    def _compute_loss(self, data):
        output= self.model_lm(dict(
            (i, j.to(self.device, non_blocking=True)) for i, j in data['x'].items() if i in ['input_ids', 'attention_mask'])
        )
        
        label= data['label'].to(self.device, non_blocking=True)
        
        if self.loss== BINARY_CROSS_ENTROPY_LOSS or self.loss== MSE_LOGARIT:
            output= output.view(-1,)
            label= label.to(dtype= torch.float32)
        return self.criterion(output, label)
    
    def _evaluate(self): 
        self.model_lm.eval()
        total_loss, total_count= 0, 0
        with torch.no_grad(): 
            for idx, data in enumerate(self.dataloader_eval): 
                loss= self._compute_loss(data)
                total_loss += loss.item() 
                if self.use_wandb: 
                    wandb.log({"Eval loss": loss.item()})
                total_count += 1 
        
        return total_loss / total_count 