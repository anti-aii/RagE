from typing import Optional, List, Type, Union
import torch, torch.nn as nn 
import numpy as np 
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType
)
import loralib as lora

from ...trainer.argument import ArgumentDataset, ArgumentTrain
from ...utils import load_model
from ...utils.process_bar import Progbar
from ..componets import selective_model_base
from ..model_rag import ModelRag
from ..model_infer import InferModel
from ...trainer.trainer import _TrainerLLM

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)


def load_model(auto_model, model_name, torch_dtype, quantization_cfg): 
    return auto_model.from_pretrained(
            model_name, 
            torch_dtype= torch_dtype, 
            device_map= 'auto' if quantization_cfg else 'cpu', 
            quantization_config= quantization_cfg
            )

class LLM(ModelRag, InferModel):
    def __init__(
        self, 
        model_name: Type[str], 
        model_base: Type[torch.nn.Module]= None, 
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]= None,
        torch_dtype= torch.float16, 
        type_backbone= "casual_lm", 
        lora_r: Optional[int]= 32, 
        lora_alpha: Optional[int]= 32, 
        lora_dropout: Optional[int]= 0.05, 
        target_modules: List[str]= None, 
        merge_lora= False, 
        gradient_ckpt: bool= True,
        use_cache: bool= False, 
        quantization_config= None, 
        torch_compile= False, 
        backend_torch_compile: str= None, 
        device= None
    ): 
        super().__init__()
        assert type_backbone in ['casual_lm', 'seq2seq']

        self.model_name= model_name
        self.lora_r= lora_r
        self.lora_alpha= lora_alpha
        self.lora_dropout= lora_dropout 
        self.target_modules= target_modules
        self.gradient_ckpt= gradient_ckpt
        self.use_cache= use_cache
        self.type_backbone= type_backbone
        self.quantization_config= quantization_config
        self.torch_compile= torch_compile
        self.backend_torch_compile= backend_torch_compile 
        self.torch_dtype= torch_dtype
        self.device= device

        # self.tokenizer= AutoTokenizer.from_pretrained(model_name, use_fast= True)

        self.model, self.tokenizer= selective_model_base(
            model_base= model_base,
            tokenizer_base= tokenizer,
            model_name= model_name, 
            type_backbone= type_backbone, 
            torch_dtype= torch_dtype, 
            quantization_config= quantization_config
        )
        
        try: 
            self.model_name= self.model.name_or_path
        except: 
            self.model_name= None 
                
        self.tokenizer.padding_side= 'left'
        
        if type_backbone== 'casual_lm': 
            self.TaskType= TaskType.CAUSAL_LM

        elif type_backbone== 'seq2seq': 
            self.TaskType= TaskType.SEQ_2_SEQ_LM

        self._setup_lora_adapter()

        self._set_dtype_device()

        self.merge_lora= merge_lora 
        self.torch_compile= torch_compile 
        
        self.__status_preprocess= False 
        

    def _setup_gradient_ckpt(self):
        # Only use for training 
        self.model.use_cache= True if self.use_cache else False
        if self.gradient_ckpt:
            self.model.gradient_checkpointing_enable()


    def _setup_lora_adapter(self): 
        config= LoraConfig(
            r= self.lora_r, 
            lora_alpha= self.lora_alpha, 
            target_modules=  self.target_modules,
            lora_dropout= self.lora_dropout, 
            bias= 'none', 
            task_type= self.TaskType
        )

        self.model= get_peft_model(self.model, config)

    def _prepare_training(self): 
        self.model.enable_input_require_grads()
        self._setup_gradient_ckpt() 

        for param in self.model.parameters():
            param.requires_grad = False  # freeze the model - train adapters later
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(torch.float32)
        self.model.lm_head= CastOutputToFloat(self.model.lm_head)        

    def compile(self, argument_train: type[ArgumentTrain], argument_dataset: type[ArgumentDataset]):
        if not self.__status_preprocess: 
            self._preprocess()
        self._prepare_training()
        self._trainer= _TrainerLLM(self, argument_train, argument_dataset)

    def _get_config_model_base(self):
        return {
            "model_type_base": self.model.__class__.__name__, 
            "model_name_or_path": self.model_name, 
            "type_backbone": self.type_backbone,
            "gradient_checkpoint": self.gradient_ckpt, 
            "use_cache": self.use_cache, 
            "quantization_config": self.quantization_config
        }

    def _get_config_addition_weight(self):
        return {
            "lora_rank": self.lora_r, 
            "lora_alpha": self.lora_alpha, 
            "lora_target_modules": self.target_modules, 
            "lora_dropout": self.lora_dropout,
        }

    def get_config(self):
        self.modules_cfg= {
            "model_base": self._get_config_model_base(),
            "lora": self._get_config_addition_weight()
            }
        return super().get_config()
    
    def forward(self, **inputs): 
        return self.model(**inputs)
    
    def _preprocess_eval(self):
        if self.training: 
            self.eval()
            
    def _preprocess(self): 
        if self.merge_lora: 
            self.model.merge_and_unload()

        if self.torch_compile:
            self._compile_with_torch()
            
        self.__status_preprocess= True

    def _preprocess_tokenize(
        self, 
        text,
        advance_config_encode: Optional[dict]= None
    ): 
        inputs= self.tokenizer.batch_encode_plus(text, padding= 'longest', return_tensors= 'pt',
                **advance_config_encode)
        return inputs
    
    def _execute_per_batch(
        self, 
        text: List[str], 
        advance_config_encode: Optional[dict]= None,
        config_generate: Optional[dict]= None
    ):
        inputs= self._preprocess_tokenize(text, advance_config_encode)
        output_sequences= self.model.generate(input_ids= inputs['input_ids'].to(self.device), 
                        attention_mask= inputs['attention_mask'].to(self.device), **config_generate)

        torch.cuda.empty_cache()
        
        return self.tokenizer.batch_decode(output_sequences, skip_special_tokens= True)
    
    def generate_text(
        self, 
        text: Union[str, List[str]], 
        batch_size= 4, 
        advance_config_encode: Optional[dict]= None,
        config_generate: Optional[dict]= None, 
        verbose= 1
    ): 
        advance_config_encode = advance_config_encode or {}
        text_output= []
        if isinstance(text, str): 
            text= [text]
        
        if not self.__status_preprocess: 
            self._preprocess() 
        self._preprocess_eval()
            
        if batch_size > len(text): 
            batch_size= len(text)
        
        batch_text= np.array_split(text, len(text)// batch_size)
        pbi= Progbar(len(text), verbose= verbose, unit_name= "Sample")
        
        for batch in batch_text: 
            text_output.extend(self._execute_per_batch(
                text, 
                advance_config_encode,
                config_generate))
            pbi.add(len(batch))
        
        return text_output

