import os 
from typing import Type, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler 
from transformers.optimization import get_cosine_schedule_with_warmup
from bitsandbytes.optim import PagedAdamW8bit
import loralib as lora 
import wandb 
from ..models import (
    CrossEncoder, 
    BiEncoder, 
    GenAnsModelCasualLM, 
    GenAnsModelSeq2SeqLM,
    GenAnsModel
)
from ..datasets import (
    GenAnsCollate, 
    GenAnsDL, 
    SentABCollate, 
    SentABDL
)


# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['WANDB_DIR'] = os.getcwd() + '/wandb/'
# os.environ['WANDB_CACHE_DIR'] = os.getcwd() + '/wandb/.cache/'
# os.environ['WANDB_CONFIG_DIR'] = os.getcwd() + '/wandb/.config/'


## default eval is loss 

class Trainer: 
    def __init__(self, 
        model, tokenizer_name: Type[str], path_datatrain: str, device: Type[torch.device], 
        path_dataeval: str= None, batch_size: int = 8, shuffle: Optional[bool]= True, num_workers: int= 16, 
        pin_memory: Optional[bool]= True, prefetch_factor: int= 8, persistent_workers: Optional[bool]= True, 
        gradient_accumlation_steps: int= 16, learning_rate: float= 1e-4, weight_decay: Optional[float]= 0.1, 
        eps: Optional[float]= 1e-6, warmup_steps: int= 150, epochs: Optional[int]= 1, path_ckpt_step: Optional[str]= 'checkpoint.pt',
        use_wandb: bool= True  
        ):

        # base 
        self.model_lm= model 
        self.tokenizer_name= tokenizer_name
        
        # dataset
        self.path_datatrain= path_datatrain
        self.path_dataeval= path_dataeval

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
        train_dataset= self.type_dataset(self.path_datatrain)
        self.dataloader_train= DataLoader(train_dataset, batch_size= self.batch_size,
                                          collate_fn= self.collate, shuffle= self.shuffle,
                                          num_workers= self.num_workers, pin_memory= self.pin_memory, 
                                          prefetch_factor= self.prefetch_factor, persistent_workers= self.persistent_workers)
        
        if self.path_dataeval: 
            eval_dataset= self.type_dataset(self.path_dataeval)
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
        self._setup_dataset()
        self._setup_optim() 
        self._setup_mxprecision()
    
    def _save_ckpt(self, param, ckpt_path= 'checkpoint.pt'): 
        return torch.save(param, ckpt_path)
    
    def _compute_loss(self, data):
        pass 

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
                if isinstance(self.model_lm, GenAnsModel):
                    self._save_ckpt({'step': idx + 1,
                                'lora': lora.lora_state_dict(self.model_lm.model),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'scheduler': self.scheduler.state_dict(),    # HERE IS THE CHANGE
                                    },  self.ckpt_step)
                else: 
                    self._save_ckpt({'step': idx + 1,
                                'model_state_dict': self.model_lm.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'scheduler': self.scheduler.state_dict(),    # HERE IS THE CHANGE
                                    },  self.ckpt_step)
        
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
            print(f'End of epoch {epoch} - loss: {train_loss}')
            print('-' * 59)
            
            if self.path_dataeval:
                print('=' * 10 + ' EVALUATE ' + '=' * 10)
                val_loss= self._evaluate()
                print(f'Evaluate loss: {val_loss}')

                if val_loss < log_loss: 
                    log_loss = val_loss
                    print(f'Saving checkpoint have best {log_loss}')
                    # torch.save(lora.lora_state_dict(model), 'bloomz_lora_domain.pt')

                    if isinstance(self.model_lm, GenAnsModel): 
                        self._save_ckpt({'epoch': epoch,
                                'lora': lora.lora_state_dict(self.mode_lm.model),
                                'scheduler': self.scheduler.state_dict(),    # HERE IS THE CHANGE
                                },  path_ckpt_epoch)
                    else: 
                        self._save_ckpt({'epoch': epoch, 
                                    'model_state_dict': self.model_lm.state_dict(), 
                                    'scheduler': self.scheduler.state_dict()}, 
                                    path_ckpt_epoch)
                        


            if train_loss < log_loss: # saving 
                log_loss = train_loss
                print(f'Saving checkpoint have best {log_loss}')
                # torch.save(lora.lora_state_dict(model), 'bloomz_lora_domain.pt')

                if isinstance(self.model_lm, GenAnsModel): 
                    self._save_ckpt({'epoch': epoch,
                            'lora': lora.lora_state_dict(self.model_lm.model),
                            'scheduler': self.scheduler.state_dict(),    # HERE IS THE CHANGE
                            },  path_ckpt_epoch)
                else: 
                    self._save_ckpt({'epoch': epoch, 
                                'model_state_dict': self.model_lm.state_dict(), 
                                'scheduler': self.scheduler.state_dict()})
        

class TrainerGenAns(Trainer):
    def __init__(self, model: Type[GenAnsModel], tokenizer_name: type[str], path_datatrain: str, 
                device: type[torch.device], path_dataeval: str = None, batch_size: int = 8, shuffle: bool | None = True, 
                 num_workers: int = 16, pin_memory: bool | None = True, prefetch_factor: int = 8, persistent_workers: bool | None = True, 
                 gradient_accumlation_steps: int = 16, learning_rate: float = 0.0001, weight_decay: float | None = 0.1, 
                 eps: float | None = 0.000001, warmup_steps: int = 150, epochs: int | None = 1, path_ckpt_step: str | None = 'checkpoint.pt', 
                 use_wandb: bool = True):
        
        super().__init__(model, tokenizer_name, path_datatrain, device, path_dataeval, 
                         batch_size, shuffle, num_workers, pin_memory, prefetch_factor, persistent_workers, gradient_accumlation_steps,
                         learning_rate, weight_decay, eps, warmup_steps, epochs, path_ckpt_step, use_wandb)
        
        self.type_dataset= GenAnsDL
        self.collate= GenAnsCollate(self.tokenizer_name)
    
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
        

class TrainerBiEncoder(Trainer):
    def __init__(self, model: Type[BiEncoder], tokenizer_name: type[str], path_datatrain: str, 
                 device: type[torch.device], path_dataeval: str = None, batch_size: int = 8, shuffle: bool | None = True, 
                 num_workers: int = 16, pin_memory: bool | None = True, prefetch_factor: int = 8, persistent_workers: bool | None = True, 
                 gradient_accumlation_steps: int = 16, learning_rate: float = 0.0001, weight_decay: float | None = 0.1, 
                 eps: float | None = 0.000001, warmup_steps: int = 150, epochs: int | None = 1, path_ckpt_step: str | None = 'checkpoint.pt', loss: str = 'cosine_embedding', 
                 use_wandb: bool = True):
        
        super().__init__(model, tokenizer_name, path_datatrain, device, path_dataeval, 
                         batch_size, shuffle, num_workers, pin_memory, prefetch_factor, persistent_workers, gradient_accumlation_steps,
                         learning_rate, weight_decay, eps, warmup_steps, epochs, path_ckpt_step, use_wandb)
        

        self.loss= loss 

        assert loss in ['sigmoid_crossentropy', 'categorical_crossentropy', 'cosine_embedding']
        if loss == 'sigmoid_crossentropy': 
            self.criterion= nn.BCEWithLogitsLoss()
        elif loss== 'categorical_crossentropy': 
            self.criterion= nn.CrossEntropyLoss(label_smoothing= 0.1)
        elif loss == 'cosine_embedding': 
            self.criterion= nn.CosineEmbeddingLoss()

        self.type_dataset= SentABDL
        self.collate= SentABCollate(self.tokenizer_name, mode= "bi_encoder")

    def _compute_loss(self, data):
        output= self.model_lm(
            dict((i, j.to(self.device, non_blocking=True)) for i, j in data['x_1'].items() if i in ['input_ids', 'attention_mask']),
            dict((i, j.to(self.device, non_blocking=True)) for i, j in data['x_2'].items() if i in ['input_ids', 'attention_mask'])
        )
        
        label= data['label'].to(self.device, non_blocking=True)
        
        if self.loss== 'sigmoid_crossentropy':
            output= output.view(-1,)
            label= label.to(dtype= torch.float32)

        loss= self.criterion(output, label)
        
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
    

class TrainerCrossEncoder(Trainer):
    def __init__(self, model:Type[CrossEncoder], tokenizer_name: type[str], path_datatrain: str, 
                 device: type[torch.device], path_dataeval: str = None, batch_size: int = 8, shuffle: bool | None = True, 
                 num_workers: int = 16, pin_memory: bool | None = True, prefetch_factor: int = 8, persistent_workers: bool | None = True, 
                 gradient_accumlation_steps: int = 16, learning_rate: float = 0.0001, weight_decay: float | None = 0.1, 
                 eps: float | None = 0.000001, warmup_steps: int = 150, epochs: int | None = 1, path_ckpt_step: str | None = 'checkpoint.pt', loss: str= 'sigmoid_crossentropy',
                 use_wandb: bool = True):
        
        super().__init__(model, tokenizer_name, path_datatrain, device, path_dataeval, 
                         batch_size, shuffle, num_workers, pin_memory, prefetch_factor, persistent_workers, gradient_accumlation_steps,
                         learning_rate, weight_decay, eps, warmup_steps, epochs, path_ckpt_step, use_wandb)
        
        self.loss= loss
        assert loss in ['sigmoid_crossentropy', 'categorical_crossentropy']
        if loss == 'sigmoid_crossentropy': 
            self.criterion= nn.BCEWithLogitsLoss()
        elif loss== 'categorical_crossentropy': 
            self.criterion= nn.CrossEntropyLoss(label_smoothing= 0.1)

        self.type_dataset= SentABDL
        self.collate= SentABCollate(self.tokenizer_name, mode= "cross_encoder")
        
    def _compute_loss(self, data):
        output= self.model_lm(dict(
            (i, j.to(self.device, non_blocking=True)) for i, j in data['x'].items() if i in ['input_ids', 'attention_mask'])
        )
        
        label= data['label'].to(self.device, non_blocking=True)
        
        if self.loss== 'sigmoid_crossentropy':
            output= output.view(-1,)
            label= label.to(dtype= torch.float32)

        loss= self.criterion(output, label)
        
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