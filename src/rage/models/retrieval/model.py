from typing import List, Type, Union, Optional
import time
import numpy as np 
import torch 
import torch.nn as nn 
import onnx 
import onnxruntime
from transformers import  AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from datasets import load_dataset

from ...trainer.argument import ArgumentDataset, ArgumentTrain
from ...trainer.trainer import _TrainerBiEncoder
from ..componets import ExtraRoberta, selective_model_base, PoolingStrategy
from ...utils import Progbar
from ...utils.convert_data import _convert_data
from ..model_rag import ModelRag
from ..model_infer import InferModel
from ...convert_to.onnx import OnnxSupport, SentenceEmbeddingOnnx




class SentenceEmbedding(ModelRag, InferModel, OnnxSupport):
    def __init__(
        self, 
        model_name= 'vinai/phobert-base-v2', 
        model_base: Type[torch.nn.Module]= None, 
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]= None, 
        type_backbone= 'mlm',
        aggregation_hidden_states= True, 
        concat_embeddings= False, 
        required_grad_base_model= True, 
        strategy_pooling= "attention_context", 
        dropout= 0.1, num_label= None, 
        torch_dtype= torch.float32, 
        device= None, 
        quantization_config= None,
        torch_compile= False, 
        backend_torch_compile: str= None
    ):

        super(SentenceEmbedding, self).__init__()

        self.model_name= model_name
        self.aggregation_hidden_states= aggregation_hidden_states
        self.concat_embeddings= concat_embeddings
        self.strategy_pooling= strategy_pooling
        self.type_backbone= type_backbone
        self.requires_grad_base_model= required_grad_base_model
        self.dropout= dropout
        self.torch_dtype= torch_dtype
        self.device= device
        self.quantization_config= quantization_config
        self.backend_torch_compile= backend_torch_compile

        # self.tokenizer= AutoTokenizer.from_pretrained(model_name, use_fast= True, add_prefix_space= True)

        self.model, self.tokenizer= selective_model_base(
            model_base= model_base, 
            tokenizer_base= tokenizer,
            model_name= model_name, 
            type_backbone= type_backbone, 
            dropout= dropout,
            using_hidden_states= aggregation_hidden_states, 
            torch_dtype= torch_dtype, 
            quantization_config= quantization_config
        )
        
        try: 
            self.model_name= self.model.name_or_path
        except: 
            self.model_name= None 
        
        self.pooling= PoolingStrategy(strategy= strategy_pooling, units= self.model.config.hidden_size)

        if not required_grad_base_model:
            self.model.requires_grad_(False)
    
        # define 
        if self.aggregation_hidden_states:
            self.extract= ExtraRoberta(method= 'mean')
        

        # dropout
        if strategy_pooling in ["attention_context", "dense_avg", "dense_first", "dense_max"]:
            self.drp1= nn.Dropout(p= dropout)
        
        # defind output 
        if self.concat_embeddings:  

            self.drp2= nn.Dropout(p= dropout)

            if not num_label: 
                self.fc= nn.Linear(self.model.config.hidden_size * 2 + 1, 128) ## suggest using with cosine similarity loss based on sentence bert paper
            else:
                self.fc= nn.Linear(self.model.config.hidden_size * 2 + 1, num_label)

            nn.init.xavier_uniform_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)

        self._set_dtype_device()

        if torch_compile:
            self._compile_with_torch()

    def get_embedding(self, input): 
        embedding= self.model(**input)

        if self.aggregation_hidden_states: 
            embedding= self.extract(embedding.hidden_states)
        else: 
            embedding= embedding.last_hidden_state

        # x= self.lnrom(embedding_enhance)
        if self.strategy_pooling in ["attention_context", "dense_avg", "dense_first", "dense_max"]: 
            embedding= self.drp1(embedding)

        x= self.pooling(embedding)

        return x
    
    def compile(
            self, 
            argument_train: type[ArgumentTrain], 
            argument_dataset: type[ArgumentDataset]
        ):
        self._trainer= _TrainerBiEncoder(self, argument_train, argument_dataset)
    
    def _get_config_model_base(self):
        return {
            "model_type_base": self.model.__class__.__name__, 
            "model_name_or_path": self.model_name, 
            "type_backbone": self.type_backbone, 
            "required_grad_base_model": self.requires_grad_base_model, 
            "aggregation_hidden_states": self.aggregation_hidden_states,
            "concat_embeddings": self.concat_embeddings,
            "dropout": self.dropout,
            "quantization_config": self.quantization_config
        }

    def  _get_config_addition_weight(self):
        return {
            "strategy_pooling": self.strategy_pooling
        }
    
    def get_config(self):
        self.modules_cfg= {
            "model_base": self._get_config_model_base(), 
            "pooling": self._get_config_addition_weight()
        }
        return super().get_config()
    
    def forward(
            self, 
            inputs, 
            return_embeddings= False
        ): 

        if return_embeddings:
            return [self.get_embedding(i) for i in inputs]
        
        if self.concat_embeddings and len(inputs) == 2: 
            x_left = self.get_embedding(inputs[0])
            x_right= self.get_embedding(inputs[1])
            x = torch.concat((x_left, x_right, torch.norm(x_right - x_left, p= 2, dim= -1).view(-1, 1)), dim= -1)
            x = self.drp2(x)
            x = self.fc(x)
            return x 
        
        raise ValueError("Input Error")
    
    def _reforward_embedding_for_onnx(self, input_ids, attention_mask): 
        inputs= {'input_ids': input_ids, 'attention_mask': attention_mask}
        return self.get_embedding(inputs)
            
    def export_onnx(self,  output_name= 'model.onnx', opset_version= 17):
        self.eval()
        example_sentence= ['This is a sentence!']
        temp_forward= self.forward
    
        self.forward= self._reforward_embedding_for_onnx
        inputs= self._preprocess_tokenize(example_sentence)
        args= {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}
        with torch.no_grad():
            torch.onnx.export(
                self, 
                args= args, 
                f= output_name, 
                input_names= ['input_ids', 'attention_mask'], 
                output_names=['output'], 
                opset_version= opset_version, 
                export_params= True,
                do_constant_folding= True, 
                # verbose= True,
                dynamic_axes= {
                    'input_ids': {0: "batch_size", 1: "seq_length"}, 
                    'attention_mask': {0: 'batch_size', 1: "seq_length"},
                    "output": {0: "batch_size"}
                }
            )
        print('**** DONE ****')
        self.forward= temp_forward
        # run check graph
        self._check_graph(output_name)
        
        # run test performance 
        dataset= load_dataset("anti-ai/ViNLI-SimCSE-supervised_v2", split="train[:1%]")['anchor']
        self._runtime_onnx(output_name= output_name)
        self._test_performance(dataset, tokenizer_function= self._preprocess_tokenize)     
    
    @classmethod
    def load_onnx(cls, model_path: str, tokenizer_name: str= 'vinai/phobert-base-v2'): 
        tokenizer= AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space= True, use_fast= True)
        return SentenceEmbeddingOnnx(model_path, tokenizer)
        
    def _preprocess(self): 
        if self.training: 
            self.eval() 
    
    def _normalize_embedding(self, input, norm:str= 'l2'): 
        assert int(norm.strip('l'))

        p= int(norm.strip('l')) 
        if p < 1: 
            raise ValueError('norm >= 1')

        norm_weight= torch.sum(input ** p, dim= -1) ** (1/p)
        return input/ norm_weight.view(-1, 1)
    
    def _preprocess_tokenize(
        self, 
        text, 
        max_legnth= 256, 
        advance_config_encode: Optional[dict]= None): 
        # 256 phobert, t5 512 
        if advance_config_encode is None: 
            advance_config_encode= dict()
            
        inputs= self.tokenizer.batch_encode_plus(text, return_tensors= 'pt', 
                padding= 'longest', max_length= max_legnth, truncation= True, 
                **advance_config_encode)
        return inputs

    def _execute_per_batch(
        self, 
        text: List[str], 
        max_length= 256, 
        advance_config_encode: Optional[dict]= None,
        normalize_embedding= None
    ):
        inputs= self._preprocess_tokenize(text, max_length, advance_config_encode)
        with torch.no_grad(): 
            embedding= self.get_embedding(dict( (i, j.to(self.device)) for i,j in inputs.items()))
        if normalize_embedding: 
            return self._normalize_embedding(embedding, norm= normalize_embedding)
        
        return embedding
    
    def encode(
        self, 
        text: Union[str, List[str]], 
        batch_size= 64, 
        max_length= 256, 
        advance_config_encode: Optional[dict]= None,
        normalize_embedding= None, 
        return_tensors= 'np', 
        verbose= 1
    ): 
        """
        Encodes the input text.

        Args:
            text (List[str]): List of input texts.
            batch_size (int): Batch size for encoding. Default is 64.
            max_length (int): Maximum length of input. Default is 256.
            normalize_embedding: Normalization method. Default is None.
            return_tensors (str): Type of output tensors. Should be one of "pt" (PyTorch tensors) or "np" (NumPy arrays). Default is 'np'.
            verbose (int): Verbosity level. Default is 1.

        Returns:
            np.ndarray or torch.Tensor: Encoded embeddings.
        """
        advance_config_encode = advance_config_encode or {}
        embeddings= []
        
        if isinstance(text, str): 
            text= [text]
        
        self._preprocess()
        if batch_size > len(text): 
            batch_size= len(text)

        batch_text= np.array_split(text, len(text)// batch_size)
        pbi= Progbar(len(text), verbose= verbose, unit_name= "Sample")

        for batch in batch_text: 
            embeddings.append(self._execute_per_batch(
                batch.tolist(), 
                max_length, 
                advance_config_encode,
                normalize_embedding))
            pbi.add(len(batch))

        return _convert_data(torch.concat(embeddings).clone().detach(), return_tensors= return_tensors)