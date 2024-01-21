from typing import Type
import random 
import yaml 

class ResponsewithRule: 
    def __init__(self, path: Type[str]): 
        # load 
        with open(path, 'r', encoding= 'utf-8') as f: 
            self.answers= yaml.full_load(f)

    def reply_begin_conversation(self): 
        return random.choice(self.answers['begin'])
    
    def reply_nonanswer(self): 
        return random.choice(self.answers['nonanswer'])
    
    def reply_close_conversation(self): 
        return random.choice(self.answers['end'])