from typing import Optional, List, Type, Union
import torch, torch.nn as nn 
from transformers import AutoModelForCausalLM
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType
)
import loralib as lora


## Only support BLOOM 

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)

class GenAnswerModel: 
    def __init__(self,
        model_name: Type[str],  device: Type[torch.device],
        torch_dtype= torch.float16, lora_r: Optional[int]= 32, 
        lora_alpha: Optional[int]= 32,  lora_dropout: Optional[int]= 0.05, 
        quantization_config= None, 
    ): 
        # initial pretrained
        self.model= AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype= torch_dtype, 
            device_map= 'cpu',  # default  
            quantization_config= quantization_config
        )

        # setup lora 
        self.lora_r= lora_r
        self.lora_alpha= lora_alpha
        self.lora_dropout= lora_dropout 

        # device 
        self.device= device

        # state 
        self._state = None # state setup lora 

    def _setup_gradient_ckpt(self, gradient_ckpt: Optional[bool]= True, use_cache: Optional[bool]= False):
        # Only use for training 
        self.model.use_cache= True if use_cache else False
        if gradient_ckpt:
            self.model.gradient_checkpointing_enable()
    
    def _setup_lora_adapter(self, lora_r: Type[int], lora_alpha: Type[int], 
                            lora_dropout: Type[float]): 
        config= LoraConfig(
            r= lora_r, 
            lora_alpha= lora_alpha, 
            target_modules= ["query_key_value"], 
            lora_dropout= lora_dropout, 
            bias= 'none', 
            task_type= TaskType.CAUSAL_LM
        )

        self.model= get_peft_model(self.model, config)


    def _prepare_training(self, gradient_ckpt: Optional[bool]= True, use_cache: Optional[bool]= False): 
        self.model.enable_input_require_grads()
        self._setup_gradient_ckpt(gradient_ckpt, use_cache) 

        for param in self.model.parameters():
            param.requires_grad = False  # freeze the model - train adapters later
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(torch.float32)
        
        self.model.lm_head= CastOutputToFloat(self.model.lm_head)

        # self._setup_lora_adapter() #setup for lora adapter 
    
    def print_trainable_parameters(self): 
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )
    
    def  return_state(self):
        return self._state
    
    def _save_ckpt(self, ckpt_dir: Optional[str]= 'ckpt_lora.pt'): 
        ## only save lora parameters 
        torch.save({'lora': lora.lora_state_dict(self.model)}, ckpt_dir)

    def _load_ckpt(self, ckpt_dir: Optional[str]= 'ckpt_lora.pt'): 
        lora_weight= torch.load(ckpt_dir, map_location= 'cpu') 
        self.model.load_state_dict(lora_weight['lora'], strict= False)
        self.model.to(self.device)

    def prepare_inference(self, ckpt_dir: Type[str], merge_lora: Optional[bool]= True, 
                          torch_compile: Optional[bool]= True, 
                          backend: Type[str]= 'onnxrt'):
        
        self._setup_lora_adapter(lora_r= self.lora_r, lora_alpha= self.lora_alpha, 
                            lora_dropout= self.lora_dropout)
        # load_lora 
        self._load_ckpt(ckpt_dir)
        if merge_lora: 
            self.model.merge_and_unload()
        if torch_compile: 
            self.model= torch.compile(self.model, backend= backend)
        
        self.model.eval() # eval mode 
        self._state= 'infer'


    def prepare_training(self, gradient_ckpt: Optional[bool]= True, use_cache: Optional[bool]= False):
        self._prepare_training(gradient_ckpt, use_cache)
        self._setup_lora_adapter(lora_r= self.lora_r, lora_alpha= self.lora_alpha, 
                                 lora_dropout= self.lora_dropout)
        self.model.to(self.device)
        self.model.train()
        self._state= 'finetune'
