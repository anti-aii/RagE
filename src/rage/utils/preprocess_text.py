import re 
from typing import Type, List
import unicodedata as ud 

class TextFormat:
    """
    A utility class for text formatting operations.

    Attributes:
        None

    Methods:
        remove_html(text: str) -> str:
            Remove HTML tags from the input text.

        remove_urls(text: str) -> str:
            Remove URLs from the input text.

        remove_emoji(text: str) -> str:
            Remove emojis from the input text.

        preprocess_text(text: str) -> str:
            Preprocess the input text by removing URLs, HTML tags, and emojis.
    """
    @staticmethod
    def remove_html(text: Type[str]):
        """
        Remove HTML tags from the input text.

        Args:
            text (str): The input text possibly containing HTML tags.

        Returns:
            str: The text with HTML tags removed.
        """
        cleanr = re.compile('<.*?>')
        return re.sub(cleanr, '', text)

    @staticmethod
    def remove_urls(text: Type[str]):
        """
        Remove URLs from the input text.

        Args:
            text (str): The input text possibly containing URLs.

        Returns:
            str: The text with URLs removed.
        """
        return re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)

    @staticmethod
    def remove_emoji(text: Type[str]):
        """
        Remove emojis from the input text.

        Args:
            text (str): The input text possibly containing emojis.

        Returns:
            str: The text with emojis removed.
        """
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
        """
        Preprocess the input text by removing URLs, HTML tags, and emojis.

        Args:
            text (str): The input text.

        Returns:
            str: The preprocessed text with URLs, HTML tags, and emojis removed.
        """
        text= ud.normalize('NFC', text.replace('\n', ' '))
        for func in [TextFormat.remove_urls, TextFormat.remove_html, TextFormat.remove_emoji]: 
            text= func(text)
        return text 