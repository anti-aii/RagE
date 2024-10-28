from typing import Optional, List, Type, Union, Iterable
import time
import numpy as np 
import onnx
import onnxruntime
import random
from ..utils.process_bar import Progbar
from ..utils.wrapper import not_allowed
from abc import abstractmethod

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class OnnxSupport: 
    def __init__(self):
        self.session_onnx= None
        self.temp_forward= None 
    
    @abstractmethod
    def export_onnx(self): 
    # only support for SentenceEmbedding and ReRanker
        raise NotImplementedError
    
    
    def _check_graph(self, output_name= 'model.onnx'): 
        model= onnx.load(output_name)
        onnx.checker.check_model(model)
        onnx.helper.printable_graph(model.graph)

    def _test_performance(self, data, tokenizer_function):
        """test with actual data  (inference time)"""
        latency= []
        print("******** Test Performance ********")
        pb= Progbar(len(data), stateful_metrics= ['time'])
        
        while len(data) > 0:
            random_number= random.randint(1, 64)
            new_data= data[:random_number]
            data= data[random_number:]
            inputs= tokenizer_function(new_data)
            start_time= time.time()
            output= self.session_onnx.run(None, {'input_ids': inputs['input_ids'].numpy(), 
                                                 'attention_mask': inputs['attention_mask'].numpy()})
            latency_r_time= time.time() - start_time
            latency.append(latency_r_time)
            values= [('time', latency_r_time)]
            pb.add(random_number, values= values)
            
        print("Average inference time = {} ms".format(format(sum(latency) * 1000 / len(latency), '.2f')))
        print("Total inference time = {} ms".format(format(sum(latency) * 1000), '.2f'))
        
    def _runtime_onnx(self, output_name= 'model.onnx'):
        self.session_onnx= onnxruntime.InferenceSession(output_name,
                    provider_options= ['CUDAExecutionProvider', 'CPUExecutionProvider'])       
    
    def run(self, data):
        self.session_onnx.run(None, data)

class SentenceEmbeddingOnnx(OnnxSupport):
    def __init__(self, session, tokenizer): 
        self.session_onnx= session
        self.tokenizer= tokenizer
        
    def _preprocess_tokenize(
        self, 
        text, 
        max_legnth= 256, 
        advance_config_encode: dict= None): 
        # 256 phobert, t5 512 
        inputs= self.tokenizer.batch_encode_plus(text, return_tensors= 'np', 
                padding= 'longest', max_length= max_legnth, truncation= True, 
                **advance_config_encode)
        return inputs

    def _normalize_embedding(self, input, norm:str= 'l2'): 
        assert int(norm.strip('l'))

        p= int(norm.strip('l')) 
        if p < 1: 
            raise ValueError('norm >= 1')

        norm_weight= np.sum(input ** p, dim= -1) ** (1/p)
        return input/ norm_weight.reshape(-1, 1)

    def _execute_per_batch(
        self, 
        text: List[str], 
        max_length= 256, 
        advance_config_encode: Optional[dict]= None,
        normalize_embedding= None
    ):
        inputs= self._preprocess_tokenize(text, max_length, advance_config_encode)
        embedding= self.run(data= {'input_ids': inputs['input_ids'].numpy(), 
                                   'attention_mask': inputs['attention_mask'].numpy()})[0]
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
        verbose= 1
    ): 
        """
        Encodes the input text.

        Args:
            text (List[str]): List of input texts.
            batch_size (int): Batch size for encoding. Default is 64.
            max_length (int): Maximum length of input. Default is 256.
            normalize_embedding: Normalization method. Default is None.
            verbose (int): Verbosity level. Default is 1.
        Returns:
            np.ndarray: Encoded embeddings.
        """
        advance_config_encode = advance_config_encode or {}
        embeddings= []
        
        if isinstance(text, str): 
            text= [text]
        
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

        return np.asarray(embeddings)
    
class ReRankerOnnx(OnnxSupport): 
    def __init__(self, session, tokenizer):
        self.session_onnx= session
        self.tokenizer= tokenizer
        
    def _preprocess_tokenize(
        self, 
        text, 
        max_legnth= 256, 
        advance_config_encode: dict= None): 
        # 256 phobert, t5 512 
        inputs= self.tokenizer.batch_encode_plus(text, return_tensors= 'np', 
                padding= 'longest', max_length= max_legnth, truncation= True, 
                **advance_config_encode)
        return inputs

    def _execute_per_batch(
        self, 
        text: List[str], 
        max_length= 256, 
        advance_config_encode: Optional[dict]= None,
    ):
        batch_text= list(map(lambda x: self.tokenizer.sep_token.join([x[0], x[1]]), text))
        inputs= self._preprocess_tokenize(batch_text, max_length, advance_config_encode)
        embedding= self.run(data= {'input_ids': inputs['input_ids'].numpy(), 
                                   'attention_mask': inputs['attention_mask'].numpy()})[0]
        return sigmoid(embedding)
    def rank(
        self, 
        text: Iterable[List[str]], 
        batch_size= 64, 
        max_length= 256, 
        advance_config_encode: Optional[dict]= None,
        verbose= 1
    ):  # [[a, b], [c, d]]
        results= [] 
        advance_config_encode = advance_config_encode or {} 

        if batch_size > len(text):
            batch_size= len(text)

        batch_text= np.array_split(text, len(text)// batch_size)
        pbi= Progbar(len(text), verbose= verbose, unit_name= "Sample")

        for batch in batch_text: 
            results.append(self._execute_per_batch(
                batch.tolist(), 
                max_length, 
                advance_config_encode))

            pbi.add(len(batch))

        return np.asarray(results)