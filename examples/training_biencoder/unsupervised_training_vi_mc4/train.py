import json  
import os 
import torch 
import wandb
from rag_chatbot import TrainerBiEncoder, BiEncoder
from rag_chatbot.utils.augment_text import NoiseDropout
from rag_chatbot.constant import (
    BINARY_CROSS_ENTROPY_LOSS,
    CATEGORICAL_CROSS_ENTROPY_LOSS,
    COSINE_SIMILARITY_LOSS,
    MSE_LOGARIT,
    CONTRASTIVE_LOSS,
    TRIPLET_LOSS, 
    IN_BATCH_NEGATIVES_LOSS
)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['WANDB_DIR'] = os.getcwd() + '/wandb/'
os.environ['WANDB_CACHE_DIR'] = os.getcwd() + '/wandb/.cache/'
os.environ['WANDB_CONFIG_DIR'] = os.getcwd() + '/wandb/.config/'

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

    with open('config.json', 'r') as f: 
        config= json.load(f)

    config_model= config['architectures']

    device= torch.device(config['device']) 

    wandb.init(
        project= "embedding-model-training",
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

    model= BiEncoder(model_name= config_model['model_name'], type_backbone= config_model['type_backbone'],
                     using_hidden_states= config_model['using_hidden_states'], concat_embeddings= config_model['concat_embeddings'],
                     required_grad= config_model['required_grad'], dropout= config_model['dropout'], 
                     hidden_dim= config_model['hidden_dim'], num_label= config_model['num_label'])

    # model.load_state_dict(torch.load('best_sentence_ckpt_zalo_grad.pt')['model_state_dict'])

    model.to(device)

    trainer= TrainerBiEncoder(model= model, tokenizer_name= config['tokenizer_name'],
                data_train= config['path_train'], max_length= config['max_length'], type_backbone= config_model['type_backbone'], 
                augment_func= NoiseDropout(), data_eval= config['path_eval'], batch_size= config['batch_size'], shuffle= config['shuffle'], 
                num_workers= config['num_workers'], pin_memory= config['pin_memory'], prefetch_factor= config['prefetch_factor'], 
                persistent_workers= config['persistent_workers'], gradient_accumlation_steps= config['gradient_accumlation_steps'],
                learning_rate= config['learning_rate'], weight_decay= config['weight_decay'], 
                eps= config['eps'], warmup_steps= config['warmup_steps'], epochs= config['epochs'],
                path_ckpt_step= config['path_ckpt_step'], device= device,  loss= loss[config['loss']], 
                )
    trainer.fit(verbose= config['verbose'], step_save= config['step_save'],
                path_ckpt_epoch= config['path_ckpt_epoch'])