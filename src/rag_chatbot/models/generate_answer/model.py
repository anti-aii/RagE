from typing import Optional, List, Type, Union
import torch, torch.nn as nn 
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType
)
import loralib as lora

### Only support Bloom and T5

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)

class GenAnsModel:
    def __init__(self,
        device: Type[torch.device], lora_r: Optional[int]= 32, 
        lora_alpha: Optional[int]= 32, lora_dropout: Optional[int]= 0.05, 
        target_modules: List[str]= None 
    ): 

        # setup lora 
        self.lora_r= lora_r
        self.lora_alpha= lora_alpha
        self.lora_dropout= lora_dropout 
        self.target_modules= target_modules 
        # device 
        self.device= device

        # state 
        self._state = None # state setup lora 

    def _setup_gradient_ckpt(self, gradient_ckpt: Optional[bool]= True, use_cache: Optional[bool]= False):
        # Only use for training 
        self.model.use_cache= True if use_cache else False
        if gradient_ckpt:
            self.model.gradient_checkpointing_enable()


    def _setup_lora_adapter(self): 
        config= LoraConfig(
            r= self.lora_r, 
            lora_alpha= self.lora_alpha, 
            target_modules=  self.target_modules, # bloom -> ["query_key_value"], t5 -> ["q", "v"] 
            lora_dropout= self.lora_dropout, 
            bias= 'none', 
            task_type= self.TaskType
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
    
    def eval(self):
        self.model.eval()
    
    def train(self):
        self.model.train()
    
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
        
        self._setup_lora_adapter()
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
        self._setup_lora_adapter()
        self.model.to(self.device)
        self.model.train()
        self._state= 'finetune'

    def gen(self, text, config_gen= None): 
        if self._state != 'infer': 
            raise 'Must use prepare_inference method.'
        
        encode= self.tokenizer.batch_encode_plus(text, padding= 'longest', 
                                                return_tensors= 'pt')
        output_sequences= self.model.generate(input_ids= encode['input_ids'].to(self.device), attention_mask= encode['attention_mask'].to(self.device), 
                                        **config_gen
                                        )
        # torch.cuda.empty_cache()
        return self.tokenizer.batch_decode(output_sequences, skip_special_tokens= True)[0] #.split('\n')[-1]     


    
        

### class GenAnsModelCausalLM  (Only support BLOOM)
class GenAnsModelCasualLM(GenAnsModel): 
    def __init__(self,
        model_name: Type[str],  device: Type[torch.device],
        torch_dtype= torch.float16, lora_r: Optional[int]= 32, 
        lora_alpha: Optional[int]= 32,  lora_dropout: Optional[int]= 0.05, 
        target_modules= ["query_key_value"], quantization_config= None,
    ): 
    
        super(GenAnsModelCasualLM, self).__init__(device, lora_r, lora_alpha, 
                                                  lora_dropout, target_modules)
        # initial pretrained
        self.model= AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype= torch_dtype, 
            device_map= 'cpu',  # default  
            quantization_config= quantization_config
        )
        self.tokenizer= AutoTokenizer.from_pretrained(model_name, use_fast= True)
        self.tokenizer.padding_side= 'left'
        self.TaskType= TaskType.CAUSAL_LM


### class GenAnsModelSeq2SeqLM  Support T5-based models
class GenAnsModelSeq2SeqLM(GenAnsModel):
    def __init__(self,
        model_name: Type[str],  device: Type[torch.device],
        torch_dtype= torch.float16, lora_r: Optional[int]= 32, 
        lora_alpha: Optional[int]= 32,  lora_dropout: Optional[int]= 0.05, 
        target_modules= ["query_key_value"], quantization_config= None,
    ): 
    
        super(GenAnsModelCasualLM, self).__init__(device, lora_r, lora_alpha, 
                                                  lora_dropout, target_modules)
        # initial pretrained
        self.model= AutoModelForSeq2SeqLM.from_pretrained(
            model_name, 
            torch_dtype= torch_dtype, # should use bf16 for training or f16 while inference 
            device_map= 'cpu',  # default  
            quantization_config= quantization_config
        )
        self.tokenizer= AutoTokenizer.from_pretrained(model_name, use_fast= True)
        self.TaskType= TaskType.SEQ_2_SEQ_LM

        
