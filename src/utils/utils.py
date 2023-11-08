import re 
from typing import Type, List
import unicodedata as ud 
import random 
import yaml 

class TextFormat:
    @staticmethod
    def remove_html(text: Type[str]):
        cleanr = re.compile('<.*?>')
        return re.sub(cleanr, '', text)

    @staticmethod
    def remove_urls(text: Type[str]):
        return re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)

    @staticmethod
    def remove_emoji(text: Type[str]):
        cleanr= re.compile("["
                            u"\U0001F600-\U0001F64F"  
                            u"\U0001F300-\U0001F5FF"
                            u"\U0001F680-\U0001F6FF"  
                            u"\U0001F1E0-\U0001F1FF"  
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
        return re.sub(cleanr, '',text)

    @staticmethod
    def preprocess_text(text: Type[str]): 
        text= ud.normalize('NFC', text.replace('\n', ' '))
        for func in [TextFormat.remove_urls, TextFormat.remove_html, TextFormat.remove_emoji]: 
            text= func(text)
        return text 
    

class ResponsewithRule: 
    def __init__(self): 
        # load 
        pass 
    def reply_begin_conversation(self): 
        pass 
    def reply_nonanswer(self): 
        pass 
    def reply_close_conversation(self): 
        pass 