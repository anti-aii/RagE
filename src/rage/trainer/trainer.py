from typing import Type, Optional, Union
from abc import abstractmethod
import pandas as pd 
import torch 
import torch.nn as nn 
import datasets
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler 
import wandb 

from .argument import ArgumentDataset, ArgumentTrain


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
    InBatchNegativeLoss,
    BINARY_CROSS_ENTROPY_LOSS,
    CATEGORICAL_CROSS_ENTROPY_LOSS,
    MSE_LOGARIT,
    COSINE_SIMILARITY_LOSS, 
    CONTRASTIVE_LOSS,
    TRIPLET_LOSS, 
    IN_BATCH_NEGATIVES_LOSS,
    EMBEDDING_RANKER_NUMERICAL,
    _criterion,
    rule_loss_task
)
from ..utils.process_bar import Progbar
from ..utils.io_utils import _print_out 
from ..utils import save_model

class _Trainer: 
    def __init__(self, model, argument_train: Type[ArgumentTrain], argument_dataset: Type[ArgumentDataset]):

        # base 
        self.model_lm= model 
        self.arg_train= argument_train
        self.arg_data= argument_dataset

        ### status 
        self.__status_setup_overall= False 


    @abstractmethod
    def _setup_dataset(self): 
        pass

    def _setup_config_argument_datasets(self): 
        self.max_length= self.arg_data.max_length
        self.tokenizer= self.arg_data.tokenizer
        self.batch_size= self.arg_data.batch_size_per_gpu * torch.cuda.device_count()
        self.shuffle= self.arg_data.shuffle
        self.num_workers= self.arg_data.num_workers
        self.pin_memory= self.arg_data.pin_memory
        self.prefetch_factor= self.arg_data.prefetch_factor
        self.persistent_workers= self.arg_data.persistent_workers
        self.augment_data_function= self.arg_data.augment_data_function

    def _setup_config_argument_train(self): 
        self.loss_function= self.arg_train.loss_function
        self.grad_accum= self.arg_train.grad_accum
        self.optimizer= self.arg_train.optimizer
        self.metrics= self.arg_train.metrics
        self.scheduler= self.arg_train.scheduler
        self.lr= self.arg_train.lr
        self.eps= self.arg_train.eps
        self.weight_decay= self.arg_train.weight_decay
        self.warmup_steps= self.arg_train.warmup_steps
        self.epochs= self.arg_train.epochs
        self.data_parallel= self.arg_train.data_parallel

    def _setup_addtion_config(self):

        raise NotImplementedError
        
    def _setup_config(self): 
        self._setup_config_argument_datasets()
        self._setup_config_argument_train()
        self._setup_addtion_config()

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
        self.optimizer= self.optimizer(self.model_lm.parameters(), self.lr, weight_decay= self.weight_decay, 
                                      eps= self.eps)
        step_epoch = len(self.dataloader_train)
        self.total_steps= int(step_epoch / self.grad_accum) * self.epochs

        self.scheduler= self.scheduler(self.optimizer, num_warmup_steps= self.warmup_steps,
                                                        num_training_steps= self.total_steps)
        
    def _setup_mxprecision(self): 
        self.scaler= GradScaler()

    def _setup_dataparallel(self): 
        if self.data_parallel: 
            self.model_lm= nn.DataParallel(self.model_lm)
        self.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
    def _setup_data(self, data_train, data_eval): 
        self.data_train= data_train 
        self.data_eval= data_eval

        self._setup_dataloader()
        self._setup_optim()

    def _setup_overall(self): 
        self._setup_config()
        self._setup_dataparallel()
        # self._setup_dataloader()
        # self._setup_optim() 
        self._setup_mxprecision()
        self.__status_setup_overall= True
    
    @abstractmethod
    def _compute_loss(self, data):
        pass 

    def _train_on_epoch(self, index_grad, verbose: int= None, step_save: int= 1000, 
        path_save_ckpt_step: str= "step_ckpt.pt", use_wandb= True): 
        self.model_lm.train()
        total_loss, total_count= 0, 0
        step_loss, step_fr= 0, 0
        
        if verbose:
            pb_i= Progbar(self.total_steps // self.epochs, verbose= verbose, stateful_metrics= self.metrics)

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

                if use_wandb:
                    wandb.log({"Train loss": step_loss / step_fr})
            
            total_loss += loss.item() 
            total_count += 1 

            if (idx + 1) % (self.grad_accum) == 0: 
                if verbose: 
                    pb_i.add(1, values= [(self.metrics, step_loss / step_fr)])
                else: 
                    _print_out(f'Step: [{index_grad[0]}/{self.total_steps}], Loss: {step_loss / step_fr}', line_break= True)
                    index_grad[0] += 1 

                step_loss = 0 
                step_fr = 0

            if (idx + 1) % step_save ==0:
                save_model(self.model_lm, path= path_save_ckpt_step, metadata= {
                    'step': idx + 1, 
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),   
                })
        
        return (total_loss / total_count) * self.grad_accum
    
    def _evaluate(self):
        pass 

    def fit(self, data_train: Union[str, pd.DataFrame, datasets.Dataset], data_eval: Union[str, pd.DataFrame, datasets.Dataset]= None, 
        verbose: int= None, use_wandb= True, step_save: int= 1000, path_save_ckpt_step: str= "step_ckpt.pt", 
        path_save_ckpt_epoch: str= "best_ckpt.pt"):
        index_grad= [1] 
        _print_out('=' * 10 + ' Setup training' + '=' * 10, line_break= True)
        
        if not self.__status_setup_overall: 
            self._setup_overall()

        self._setup_data(data_train, data_eval)        

        _print_out('=' * 10 + ' Begin training ' + '=' * 10, line_break= True)
        log_loss = 1e9
        for epoch in range(1, self.epochs + 1): 
            print("\nEpoch {}/{}".format(epoch, self.epochs))

            train_loss= self._train_on_epoch(index_grad, verbose, step_save, path_save_ckpt_step= path_save_ckpt_step, 
                                        use_wandb= use_wandb)
            # val_loss = evaluate()

            _print_out('-' * 59 + f'\nEnd of epoch {epoch} - loss: {train_loss}\n' + '-' * 59, line_break= True)
            
            if self.data_eval:
                _print_out('=' * 10 + ' EVALUATE ' + '=' * 10, line_break= True)
                val_loss= self._evaluate()
                _print_out(f'Evaluate loss: {val_loss}', line_break= True)

                if val_loss < log_loss: 
                    log_loss = val_loss
                    _print_out(f'Saving checkpoint have best {log_loss}', line_break= True)
                    save_model(self.model_lm, path= path_save_ckpt_epoch)
                        
            if train_loss < log_loss: # saving 
                log_loss = train_loss
                _print_out(f'Saving checkpoint have best {log_loss}', line_break= True)
                save_model(self.model_lm, path= path_save_ckpt_epoch)
        

class _TrainerLLM(_Trainer):
    def __init__(self, model, argument_train: Type[ArgumentTrain], argument_dataset: Type[ArgumentDataset]):
        super().__init__(model, argument_train= argument_train, argument_dataset= argument_dataset)
        
    def _setup_addtion_config(self):
        self.collate= GenAnsCollate(self.tokenizer, self.max_length)

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
        

class _TrainerBiEncoder(_Trainer):  ## support 
    def __init__(self, model, argument_train: Type[ArgumentTrain], argument_dataset: Type[ArgumentDataset]):
        
        super().__init__(model, argument_train= argument_train, argument_dataset= argument_dataset)
        
    def _setup_addtion_config(self):
        self.task= rule_loss_task(self.loss_function)
        
        if isinstance(self.loss_function, str): 

            assert self.loss_function in [BINARY_CROSS_ENTROPY_LOSS, 
                            CATEGORICAL_CROSS_ENTROPY_LOSS, 
                            COSINE_SIMILARITY_LOSS, 
                            CONTRASTIVE_LOSS,
                            TRIPLET_LOSS,
                            IN_BATCH_NEGATIVES_LOSS]
            self.criterion= _criterion[self.loss_function]

        else: 
            assert isinstance(self.loss_function, (nn.BCEWithLogitsLoss, nn.CrossEntropyLoss,
                            CosineSimilarityLoss, ContrastiveLoss, TripletLoss, InBatchNegativeLoss))
            
            self.criterion= self.loss_function
        
        self.collate= SentABCollate(self.tokenizer, mode= "bi_encoder",
                    max_length= self.max_length, task= self.task, augment_func= self.augment_data_function)
    
    def _setup_dataset(self):
        train_dataset= SentABDL(self.data_train, task= self.task)

        if self.data_eval: 
            return train_dataset, SentABDL(self.data_eval, task= self.task)
        
        return train_dataset, None

    def _compute_loss(self, data):        
        if self.task != EMBEDDING_RANKER_NUMERICAL: 

            if self.loss_function == CONTRASTIVE_LOSS or isinstance(self.loss_function, ContrastiveLoss): 
                output= self.model_lm(
                    (
                    dict((i, j.to(self.device, non_blocking=True)) for i, j in data['sent1'].items() if i in ['input_ids', 'attention_mask']),
                    dict((i, j.to(self.device, non_blocking=True)) for i, j in data['sent2'].items() if i in ['input_ids', 'attention_mask'])
                    ),
                    return_embeddings= True
                )
                return self.criterion(output[0], output[1], data['label'])
            

            if self.loss_function == TRIPLET_LOSS or isinstance(self.loss_function, TripletLoss): 
                output= self.model_lm(
                    (
                    dict((i, j.to(self.device, non_blocking=True)) for i, j in data['anchor'].items() if i in ['input_ids', 'attention_mask']),
                    dict((i, j.to(self.device, non_blocking=True)) for i, j in data['pos'].items() if i in ['input_ids', 'attention_mask']),
                    dict((i, j.to(self.device, non_blocking=True)) for i, j in data['neg'].items() if i in ['input_ids', 'attention_mask'])
                    ),
                    return_embeddings= True
                )
                return self.criterion(*output)
            
            if self.loss_function == IN_BATCH_NEGATIVES_LOSS or isinstance(self.loss_function, InBatchNegativeLoss): 
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

            if self.loss_function== COSINE_SIMILARITY_LOSS or isinstance(self.loss_function, CosineSimilarityLoss): 
                output= self.model_lm(data_input, return_embeddings= True)
                return self.criterion(output[0], output[1], label.to(dtype= torch.float32))
            
            output= self.model_lm(data_input)

            if self.loss_function== BINARY_CROSS_ENTROPY_LOSS or isinstance(self.loss_function, nn.BCEWithLogitsLoss):
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
    
    
class _TrainerCrossEncoder(_Trainer):
    def __init__(self, model, argument_train: Type[ArgumentTrain], argument_dataset: Type[ArgumentDataset]):
        
        super().__init__(model, argument_train= argument_train, argument_dataset= argument_dataset)
        
    def _setup_addtion_config(self):
        self.task= rule_loss_task(self.loss_function)

        if isinstance(self.loss_function, str): 

            assert self.loss_function in [BINARY_CROSS_ENTROPY_LOSS, 
                            CATEGORICAL_CROSS_ENTROPY_LOSS, 
                            MSE_LOGARIT]
            self.criterion= _criterion[self.loss_function]

        else: 
            assert isinstance(self.loss_function, (nn.BCEWithLogitsLoss, nn.CrossEntropyLoss,
                            MSELogLoss))
            
            self.criterion= self.loss_function
    
        self.collate= SentABCollate(self.tokenizer, mode= "cross_encoder", 
                    max_length= self.max_length, task= self.task, augment_func= self.augment_data_function)
        
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
        
        if (self.loss_function== BINARY_CROSS_ENTROPY_LOSS or isinstance(self.loss_function, 
                nn.BCEWithLogitsLoss)) or (self.loss_function== MSE_LOGARIT or isinstance(self.loss_function, MSELogLoss)):
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