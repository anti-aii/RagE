import torch
import onnxruntime
from abc import abstractmethod

class OnnxSupport: 
    def __init__(self):
        self.status_load_onnx= False
        self.temp_forward= None 
    
    @abstractmethod
    def export_onnx(self): 
    # only support for SentenceEmbedding and ReRanker
        raise NotImplementedError
    
    def print_graph(self): 
        pass 
        
    def _runtime_onnx(self):
        pass
    
    @abstractmethod
    def run(self):
        raise NotImplementedError
    
    