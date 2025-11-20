from abc import ABC, abstractmethod
from pydantic import BaseModel


class LlmInterface(ABC):
    @abstractmethod
    def invoke(self, prompt: str, structure_schema:BaseModel=None) -> BaseModel | str:
        pass
    

    @abstractmethod
    def invoke_stream(self, prompt: str):
        pass
    

    @abstractmethod
    def set_system_prompt(self, system_prompt: str):
        pass


    @abstractmethod
    def set_temperature(self, temperature: int):
        pass