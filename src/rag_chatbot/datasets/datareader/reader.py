from typing import Type 
import pandas as pd 

class DataReader: 
    def __init__(self, filename: Type[str]): 
        """
        Initializes a new instance of the domain_reader class.

        Args:
        filename (str): The name of the file to read.
        """
        self.__filename= filename
        self.__data= None 
        assert filename.endswith(".csv") or filename.endswith(".json"), "Currently, We only support two type formats: csv and json and txt"
    
    def __raise_empty(self):
        if self.__data is None:
            raise Exception("Data is Empty!")
        
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

        if self.__filename.endswith(".csv"):
            self._read_csv()
        elif self.__filename.endswith(".json"):
            self._read_json()
        elif self.__filename.endswith(".txt"):
            self._read_txt()

        assert len(self.__data.columns) == 3, "The number of columns must be 3"

        return self.__data 
