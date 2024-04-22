from abc import abstractmethod, ABC

class InferModel(ABC): 
    
    @abstractmethod
    def _preprocess(self):
        pass 

    @abstractmethod
    def _preprocess_tokenize(self): 
        pass 
    
    @abstractmethod
    def _execute_per_batch(self):
        pass