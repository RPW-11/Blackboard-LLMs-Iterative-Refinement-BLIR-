from abc import ABC, abstractmethod


class BlackboardInterface(ABC):
    def __init__(self, storage_path:str):
        self.storage_path = storage_path

    
    @abstractmethod
    def save_attempt(self, data):
        pass
    
    
    @abstractmethod
    def get_attempt_results(self, n:int = 10) -> str:
        pass

    