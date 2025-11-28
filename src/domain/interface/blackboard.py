from abc import ABC, abstractmethod
from typing import List


class BlackboardInterface(ABC):
    def __init__(self, storage_path:str):
        self.storage_path = storage_path

    
    @abstractmethod
    def save_attempt(self, data, file_name: str = None):
        pass
    
    
    @abstractmethod
    def get_attempt_results(self, n:int = 10) -> List[dict]:
        pass

    