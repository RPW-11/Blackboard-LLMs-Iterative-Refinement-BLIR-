from abc import ABC, abstractmethod
from pydantic import BaseModel


class Problem(ABC):
    def __init__(self, description: str):
        self.description = description

    @abstractmethod
    def get_best_result(self)->dict:
        pass

    @abstractmethod
    def apply_llm_response(self, response:BaseModel):
        pass