import json 
import sys 
import os 
from argparse import ArgumentParser
import torch 
import wandb
from rag_chatbot import TrainerCrossEncoder, SentABDL, SentABCollate, CrossEncoder


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['WANDB_DIR'] = os.getcwd() + '/wandb/'
os.environ['WANDB_CACHE_DIR'] = os.getcwd() + '/wandb/.cache/'
os.environ['WANDB_CONFIG_DIR'] = os.getcwd() + '/wandb/.config/'

wandb.login()

if __name__ == "__main__": 

    with open('config.json', 'r') as f: 
        config= json.load(f)

    device= torch.device(config['device']) 

    wandb.init(
        project= "ranker-utt",
        name= config['name'], 
        config= {
            'model': config['model_name'], 
            'required_grad': config['required_grad'],
            'num_label': config['num_label'],
            'loss': config['loss'],
            'batch_size': config['batch_size'],
            'epochs': config['epochs'], 
            'lr': config['learning_rate'],
            'data': config['path_train']
        }
    )

    model= CrossEncoder(model_name= config['model_name'], required_grad= config['required_grad'], num_label= config['num_label'])
    model.to(device)

    trainer= TrainerCrossEncoder(model= model, tokenizer_name= config['tokenizer_name'],
                path_datatrain= config['path_train'], path_dataeval= config['path_eval'],
                batch_size= config['batch_size'], shuffle= config['shuffle'], num_workers= config['num_workers'],
                pin_memory= config['pin_memory'], prefetch_factor= config['prefetch_factor'], 
                persistent_workers= config['persistent_workers'], gradient_accumlation_steps= config['gradient_accumlation_steps'],
                learning_rate= config['learning_rate'], weight_decay= config['weight_decay'], 
                eps= config['eps'], warmup_steps= config['warmup_steps'], epochs= config['epochs'],
                path_ckpt_step= config['path_ckpt_step'], device= device,  loss= config['loss'], 
                )
    trainer.fit(verbose= config['verbose'], step_save= config['step_save'],
                path_ckpt_epoch= config['path_ckpt_epoch'])