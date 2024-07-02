from typing import Type, Union, Iterable, Dict
from abc import abstractmethod
import pandas as pd 
import torch 
from torch import nn, Tensor
import datasets
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler 
import wandb 

from .rag_parallel import RagDataParallel
from .argument import ArgumentDataset, ArgumentTrain


from ..datasets import (
    GenAnsCollate, 
    GenAnsDL, 
    SentABCollate, 
    SentABDL
)
from ..losses import (
    LossRAG, 
    CosineSimilarityLoss, 
    MSELogLoss, 
    ContrastiveLoss,
    TripletLoss, 
    InBatchNegativeLoss,
    GITSEmbedLoss,
    BinaryCrossEntropy, 
    CategoricalCrossEntropy
)

from ..constant import (
    EMBEDDING_CONTRASTIVE, 
    EMBEDDING_IN_BATCH_NEGATIVES, 
    EMBEDDING_RANKER_NUMERICAL, 
    EMBEDDING_TRIPLET
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
        self.advance_config_encode= self.arg_data.advance_config_encode
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
            self.model_lm= RagDataParallel(self.model_lm)
        self.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model_lm.to(self.device)
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

    def _take_future(self, features: Iterable[Dict[str, Tensor]], labels: Tensor= None): 
        # input: 
        result= []
        for value in features: 
            result.append(dict((i, j.to(self.device, non_blocking=True)) for i, j in value.items()))

        if not labels is None: 
            return {'features': result, 
                    'labels': labels.to(self.device, non_blocking= True)
            }
        else:
            return {'features': result}
    
    def _select_loss_functon(self, loss_function: Union[str, LossRAG]) -> LossRAG: 
        if isinstance(loss_function, str):
            ## defind list loss defaul param 
            list_loss= [
            CosineSimilarityLoss(), 
            MSELogLoss(), 
            ContrastiveLoss(),
            TripletLoss(), 
            InBatchNegativeLoss(),
            CategoricalCrossEntropy(),
            BinaryCrossEntropy()
            ]

            for loss_func in list_loss: 
                if loss_function == loss_func.pretty_name: 
                    loss_func.compile(model= self.model_lm)
                    return loss_func
            raise ValueError("The loss function you entered does not exist or is not supported to use with pretty_name.\
Please read the supported loss functions carefully")

        else: 
            try: 
                loss_function.compile(model= self.model_lm)
                return loss_function
            except: 
                raise ValueError("If you use custom loss functions, you must inherit the LossRAG found in rage.losses.LossRAG")
            
    
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
        self.collate= GenAnsCollate(self.model_lm.tokenizer, self.max_length, self.advance_config_encode)

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
        self.loss= self._select_loss_functon(self.loss_function)
        
        self.collate= SentABCollate(self.model_lm.tokenizer, mode= "bi_encoder", advance_config_encode= self.advance_config_encode,
                    max_length= self.max_length, task= self.loss.task_name, augment_func= self.augment_data_function)
    
    def _setup_dataset(self):
        train_dataset= SentABDL(self.data_train, task= self.loss.task_name)

        if self.data_eval: 
            return train_dataset, SentABDL(self.data_eval, task= self.loss.task_name)
        
        return train_dataset, None

    def _compute_loss(self, data):        
        if self.loss.task_name == EMBEDDING_CONTRASTIVE:
            data= self._take_future(
                features= [
                    data['sent1'], data['sent2']
                ], 
                label= data['label']
            )
            
        elif self.loss.task_name == EMBEDDING_TRIPLET: 
            data= self._take_future(
                features= [
                    data['anchor'], data['pos'], data['neg']
                ]
            )
        
        elif self.loss.task_name  == EMBEDDING_IN_BATCH_NEGATIVES: 
            mini_data=[data['anchor'], data['pos'], data['neg']] 
            if len(data) > 2: 
                for i in range(len(data)-2): 
                    mini_data.append(data[f'hard_neg_{i+1}'])

            data= self._take_future(
                features= mini_data
            )
            
        elif self.loss.task_name == EMBEDDING_RANKER_NUMERICAL: 
            data= self._take_future(
                features= [
                    data['x_1'], data['x_2']
                ], 
                labels= data['label']
            )
        else: 
            raise ValueError(
                "If you run custom loss functions, you must attach a task_name to it.\
                 Please refer to the loss functions defined in rage.losses.")
            
        return self.loss(**data) 
            
    
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
        self.loss= self._select_loss_functon(self.loss_function)
        
        self.collate= SentABCollate(self.model_lm.tokenizer, mode= "cross_encoder", advance_config_encode= self.advance_config_encode,
                    max_length= self.max_length, task= self.loss.task_name, augment_func= self.augment_data_function)
        
    def _setup_dataset(self):
        train_dataset= SentABDL(self.data_train, task= self.loss.task_name)

        if self.data_eval: 
            return train_dataset, SentABDL(self.data_eval, task= self.loss.task_name)
        
        return train_dataset, None

    def _compute_loss(self, data):
        if self.loss.task_name== EMBEDDING_RANKER_NUMERICAL:
            data= self._take_future(
                features= [
                    data['x']
                ], 
                labels= data['label']
            )
        
        return self.loss(data['features'][0], data['labels'])
    
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