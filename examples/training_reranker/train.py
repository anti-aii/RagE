import json  
import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

import torch 
import wandb
from datasets import load_dataset
from transformers.optimization import get_cosine_schedule_with_warmup
from bitsandbytes.optim import PagedAdamW8bit


from rage import Reranker, ArgumentDataset, ArgumentTrain

from rage.losses import (
    BINARY_CROSS_ENTROPY_LOSS,
    CATEGORICAL_CROSS_ENTROPY_LOSS,
    COSINE_SIMILARITY_LOSS,
    MSE_LOGARIT,
    CONTRASTIVE_LOSS,
    TRIPLET_LOSS, 
    IN_BATCH_NEGATIVES_LOSS
)

wandb.login()

loss= {
    'binary_cross_entropy': BINARY_CROSS_ENTROPY_LOSS, 
    'categorical_cross_entropy': CATEGORICAL_CROSS_ENTROPY_LOSS, 
    'cosine_similarity': COSINE_SIMILARITY_LOSS, 
    'mse_log': MSE_LOGARIT, 
    'contrastive': CONTRASTIVE_LOSS, 
    'triplet': TRIPLET_LOSS, 
    'in_batch_negatives': IN_BATCH_NEGATIVES_LOSS

}

if __name__ == "__main__": 
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("Available GPUs:", torch.cuda.device_count())

    with open('config.json', 'r') as f: 
        config= json.load(f)

    config_model= config['architectures']

    device = torch.device(config['device']) 

    wandb.init(
        project= "cross_encoder-training",
        name= config['name'], 
        config= {
            'model': config_model['model_name'], 
            'required_grad': config_model['required_grad'],
            "dropout": config_model['dropout'], 
            "using_hidden_states": config_model['using_hidden_states'], 
            "pooling": config_model['pooling'], 
            'max_length': config['max_length'],
            'loss': config['loss'],
            'batch_size': config['batch_size'],
            'epochs': config['epochs'], 
            'lr': config['learning_rate'],
            'warmup_steps': config['warmup_steps'],
            'data': config['path_train']
        }
    )

    model= Reranker(
            model_name= config_model['model_name'], 
            type_backbone= config_model['type_backbone'],
            aggregation_hidden_states= config_model['using_hidden_states'], 
            required_grad_base_model= config_model['required_grad'], 
            dropout= config_model['dropout'], 
            strategy_pooling= config_model['pooling'],
            num_label= config_model['num_label']
    )

    model.to(device)
    
    args_train= ArgumentTrain(
                loss_function= loss[config['loss']],
                gradient_accumlation_steps= config['gradient_accumlation_steps'], 
                learning_rate= config['learning_rate'], 
                weight_decay= config['weight_decay'], 
                eps= config['eps'], 
                warmup_steps= config['warmup_steps'],
                epochs= config['epochs'], 
                optimizer= PagedAdamW8bit, 
                metrics= "loss", 
                scheduler= get_cosine_schedule_with_warmup, 
                data_parallel= True
    )
    
    ars_dataset= ArgumentDataset(
                max_length= config['max_length'], 
                batch_size_per_gpu= config['batch_size'], 
                shuffle= config['shuffle'], 
                augment_data_function= None,
                pin_memory= config['pin_memory'], 
                num_workers= config['num_workers'],
                prefetch_factor= config['prefetch_factor'],
                persistent_workers= config['persistent_workers'],
    )

    model.compile(args_train, ars_dataset)

    model.fit(data_train= config['path_train'], 
            data_eval= config['path_eval'], 
            path_save_ckpt_step= config['path_ckpt_step'], 
            path_save_ckpt_epoch= config['path_ckpt_epoch'], 
            step_save= config['step_save'],
            verbose= config['verbose']
    )

