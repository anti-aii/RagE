from typing import Union
import datasets
import pandas as pd 

## reader frin file 
class DataReader: 
    def __init__(self, path_or_data: Union[str, pd.DataFrame, datasets.Dataset], condition_func= None): 
        self.__filename= None 
        self.__data= None 

        if isinstance(path_or_data, pd.DataFrame) or isinstance(path_or_data, datasets.Dataset): 
            self.__data= path_or_data 
        elif isinstance(path_or_data, str): 
            self.__filename= path_or_data 

        self.__condition_func= condition_func
        
        if self.__filename: 
            assert self.__filename.endswith(".csv") or self.__filename.endswith(".json"), "Currently, We only support three type formats: csv and json and txt"
    
    def __raise_empty(self):
        if self.__data is None:
            raise ValueError("Data is Empty!")
        
    def __len__(self):
        self.__raise_empty()
        return len(self.__data)
        
    def _read_csv(self): 
        """
        Reads the data from a csv file.
        """
        self.__data= pd.read_csv(self.__filename)

    def _read_json(self):
        """
        Reads the data from a json file.
        """
        self.__data= pd.read_json(self.__filename)
    
    def _read_txt(self):
        """
        Reads the data rom a text file 
        """
        self.__data= pd.read_csv(self.__filename, delimiter= '|', header= None)

    def read(self): 
        if self.__filename:
            if self.__filename.endswith(".csv"):
                self._read_csv()
            elif self.__filename.endswith(".json"):
                self._read_json()
            elif self.__filename.endswith(".txt"):
                self._read_txt()
            else:
                raise ValueError("Only support .txt, .json, .csv formart")

        # assert len(self.__data.columns) == 3, "The number of columns must be 3"
        
        if self.__condition_func:
            assert callable(self.__condition_func)

            return self.__condition_func(self.__data)
        return self.__data     
