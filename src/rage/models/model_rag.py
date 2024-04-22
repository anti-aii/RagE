import torch 
import os 
import json 
import datetime
import pandas as pd 
import datasets 
from pathlib import Path
from abc import abstractmethod
from typing import Type, Union
from prettytable import PrettyTable 

from huggingface_hub import HfApi, hf_hub_download, ModelCard, ModelCardData
from huggingface_hub.utils import (
    EntryNotFoundError, 
    HfHubHTTPError,
    SoftTemporaryDirectory,
    is_torch_available,
    validate_hf_hub_args,
)
from huggingface_hub.utils._deprecation import _deprecate_arguments

from ..trainer.argument import ArgumentDataset, ArgumentTrain
from ..utils import (
    save_model, 
    load_model, 
)
from ..utils.io_utils import _create_new_path, _ensure_dir, _print_out
from ..utils.print_trainable_params import _count_params_of_model
from ..constant import CONFIG_MODEL, PYTORCH_WEIGHTS_SAVE_PRETRAIN


import logging 
logger= logging.Logger(__name__)

class ModelRag(torch.nn.Module): 
    ### currently support for BiEncoder, CrossEncoder
    @abstractmethod
    def _get_config_model_base(self):
        # return architecture base model, 
        pass 
    
    @abstractmethod
    def _get_config_addition_weight(self): 
        pass 
    
    # @abstractmethod
    def get_config(self):
        # return dict(self._get_config_model_base, **self._get_config_addition_weight)
        return  {
            "architecture": self.__class__.__name__, 
            "modules": self.modules_cfg
        }
    
    def _config2str(self): 
        config_name= str(self.__class__.__name__) + "Config("

        for key, value in self.get_config()['modules'].items(): 
            config_name += f"{key}: {value}, "
        
        config_name= config_name.rstrip(', ')
        config_name += ")"

        return config_name
    
    def _set_dtype_device(self): 
        self.to(self.device, dtype= self.torch_dtype)
    
    def _compile_with_torch(self): 
        self.forward= torch.compile(self.forward, fullgraph= True, mode= "reduce-overhead",
                                    backend= self.backend_torch_compile)

    
    def _save_config(self, file_name:str): 
        with open(file_name, 'w', encoding= 'utf-8') as f: 
            json.dump(self.get_config(), f, ensure_ascii= False)
    
    def save(self, path: str, mode: str= "trainable_weight", limit_size= 6,
               size_limit_file= 3, storage_units= 'gb', key:str= 'model_state_dict', metada: dict= None):
        
        current_time= datetime.datetime.now()
        folder_name= current_time.strftime("%Y-%m-%d_%H-%M-%S")

        path= _create_new_path(path, folder_name)
        # path= os.path.join(path, folder_name)

        save_model(self, path, mode, limit_size, size_limit_file, storage_units, key,
                   metada)
        
        path_config, _= os.path.split(path)
        self._save_config(file_name= os.path.join(path_config, CONFIG_MODEL))
    
    def save_pretrained(
            self, 
            save_directory: str, 
            *, 
            config: dict, 
            repo_id= None, 
            push_to_hub: False,
            **push_to_hub_kwargs): 
        
        _ensure_dir(save_directory, create_path= True)

        ## save weight
        path_weight= os.path.join(save_directory, PYTORCH_WEIGHTS_SAVE_PRETRAIN)
        save_model(self, path_weight, key= None)

        ## save config 
        self._save_config(file_name= os.path.join(save_directory, CONFIG_MODEL))

        if push_to_hub: 
            kwargs= push_to_hub_kwargs.copy()
            if config is not None: 
                kwargs['config']= self.get_config()
            
            if repo_id is None: 
                repo_id= os.path.split(save_directory)[1]

            return self.push_to_hub(repo_id= repo_id, **kwargs)
    
    @classmethod
    @validate_hf_hub_args
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path, 
        *, 
        force_download: bool= False, 
        resume_download: bool= False, 
        proxies= None,  
        token= None,
        cache_dir= None,
        local_files_only= False,
        revision= None,
        **model_kwargs,
    ): 
        model_id = str(pretrained_model_name_or_path)
        config_file= None 

        if os.path.isdir(model_id):
            if CONFIG_MODEL in os.listdir(model_id):
                config_file= os.path.join(model_id, CONFIG_MODEL)
            else: 
                logger.warning(f"{CONFIG_MODEL} not found in {Path(model_id).resolve()}")
        
        else: 
            try: 
                config_file= hf_hub_download(
                    repo_id= model_id, 
                    filename= CONFIG_MODEL, 
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e: 
                logger.info(f"{CONFIG_MODEL} not found on the HuggingFace Hub: {str(e)}")
        
        config= None 
        if config_file is not None: 
            with open(config_file, "r", encoding= "utf-8") as f: 
                config= json.load(f)

            new_config= dict()

            for key in config['modules']: 
                new_config.update(config['modules'][key])
            new_config.pop('model_type_base')
            new_config.update(model_kwargs)
            model_kwargs = new_config

        model= cls(**model_kwargs)

        if os.path.isdir(model_id): 
            _print_out("Loading weights from local directory")
            model_file= os.path.join(model_id, PYTORCH_WEIGHTS_SAVE_PRETRAIN)
            model.load(model_file, key= None)

        else:
            model_file= hf_hub_download(
                repo_id= model_id, 
                filename= PYTORCH_WEIGHTS_SAVE_PRETRAIN, 
                revision= revision, 
                cache_dir= cache_dir, 
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )

            model.load(model_file, key= None)
        
        return model 
    
    @validate_hf_hub_args
    def push_to_hub(
        self, 
        repo_id: str, 
        *, 
        config= None, 
        commit_message: str = "Push model using huggingface_hub.",
        private: bool = False,
        token: str = None,
        branch: str = None,
        create_pr: bool = None,
        allow_patterns = None,
        ignore_patterns= None,
        delete_patterns= None,
        # TODO: remove once deprecated
        api_endpoint:str = None,
    ): 
        api= HfApi(endpoint= api_endpoint, token= token)
        repo_id= api.create_repo(repo_id= repo_id, private= private, exist_ok= True).repo_id

        with SoftTemporaryDirectory() as tmp: 
            saved_path= os.path.join(tmp, repo_id)
            self.save_pretrained(saved_path, config= config)
            return api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=saved_path,
                commit_message=commit_message,
                revision=branch,
                create_pr=create_pr,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                delete_patterns=delete_patterns,
            )

    def load(self, path: str, multi_ckpt= False, key: str= 'model_state_dict'): 
        load_model(self, path, multi_ckpt, key)
        self.eval()

    def summary_params(self): 
        _count_params_of_model(self, count_trainable_params= True, 
                              return_result= False)

    def summary(self): 
        table= PrettyTable(['Layer (type)', 'Params', 'Trainable params'])
        for name, weight in self.named_children(): 
            count_params= _count_params_of_model(weight, count_trainable_params= True, return_result= True)
            table.add_row([f"{name} ({weight.__class__.__name__})", f"{count_params['all_params']:,}", f"{count_params['trainable_params']}"])
        
        print(table)
    
    @abstractmethod
    def compile(self, argument_train: Type[ArgumentTrain], argument_dataset: Type[ArgumentDataset]):
        pass

    def fit(
        self, 
        data_train: Union[str, pd.DataFrame, datasets.Dataset], 
        data_eval: Union[str, pd.DataFrame, datasets.Dataset]= None,
        verbose= 1, 
        use_wandb= True, 
        step_save: int= 1000, 
        path_save_ckpt_step: str= "step_ckpt.pt", 
        path_save_ckpt_epoch: str= "best_ckpt.pt"
    ):
        
        self._trainer.fit(
            data_train, 
            data_eval, 
            verbose, 
            use_wandb, 
            step_save, 
            path_save_ckpt_step, 
            path_save_ckpt_epoch
        )


    def evaluation(self, data):
        # currently not support 
        pass 
    
    @abstractmethod
    def _preprocess(self):
        pass 

    @abstractmethod
    def _preprocess_tokenize(self): 
        pass 
    
    @abstractmethod
    def _execute_per_batch(self):
        pass

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self):
        return self._config2str()
