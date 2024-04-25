import json  
import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

import torch 
import wandb
from datasets import load_dataset
from transformers.optimization import get_cosine_schedule_with_warmup
from bitsandbytes.optim import PagedAdamW8bit


from rage import LLM, ArgumentDataset, ArgumentTrain, NoiseMask

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


if __name__ == "__main__": 
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("Available GPUs:", torch.cuda.device_count())

    with open('config.json', 'r') as f: 
        config= json.load(f)

    config_model= config['architectures']

    device = torch.device(config['device']) 

    wandb.init(
        project= "fine_tune_llm_qa_abtract",
        name= config['name'], 
        config= {
            'model': config_model['model_name'], 
            'lora_r': config_model['lora_r'], 
            'lora_alpha': config_model['lora_alpha'], 
            'lora_dropout': config_model['lora_dropout'], 
            'target_modules': config_model['target_modules'], 
            'merge_lora': config_model['merge_lora'], 
            'gradient_ckpt': config_model['gradient_ckpt'], 
            'use_cache': config_model['use_cache'], 
            'quantization_config': None,
            'max_length': config['max_length'],
            'batch_size': config['batch_size'],
            'epochs': config['epochs'], 
            'lr': config['learning_rate'],
            'warmup_steps': config['warmup_steps'],
            'data': config['path_train']
        }
    )

    model= LLM(
            model_name= config_model['model_name'], 
            type_backbone= config_model['type_backbone'],
            lora_r= config_model['lora_r'], 
            lora_alpha= config_model['lora_alpha'], 
            lora_dropout= config_model['lora_dropout'], 
            target_modules= config_model['target_modules'], 
            merge_lora= config_model['merge_lora'], 
            gradient_ckpt= config_model['gradient_ckpt'], 
            use_cache= config_model['use_cache'], 
            quantization_config= None,
    )

    model.to(device)
    
    args_train= ArgumentTrain(
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
                tokenizer= config['tokenizer_name'],
                max_length= config['max_length'], 
                batch_size_per_gpu= config['batch_size'], 
                shuffle= config['shuffle'], 
                augment_data_function= NoiseMask(0.3),
                pin_memory= config['pin_memory'], 
                num_workers= config['num_workers'],
                prefetch_factor= config['prefetch_factor'],
                persistent_workers= config['persistent_workers'],
    )

    model.compile(args_train, ars_dataset)

    model.fit(data_train= load_dataset(config['path_train'], split= "train"), 
            data_eval= config['path_eval'], 
            path_save_ckpt_step= config['path_ckpt_step'], 
            path_save_ckpt_epoch= config['path_ckpt_epoch'], 
            step_save= config['step_save']
    )

