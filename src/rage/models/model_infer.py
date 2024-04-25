from abc import abstractmethod, ABC

class InferModel(ABC): 
    
    @abstractmethod
    def _preprocess(self):
        raise NotImplementedError

    @abstractmethod
    def _preprocess_tokenize(self): 
        raise NotImplementedError
    
    @abstractmethod
    def _execute_per_batch(self):
        raise NotImplementedError