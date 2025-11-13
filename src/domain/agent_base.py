from .interface.llm import LlmInterface
from abc import ABC, abstractmethod


class AgentBase(ABC):
    def __init__(self, llm: LlmInterface, system_prompt_path: str):
        super().__init__()
        self.llm = llm

        try:
            self.system_prompt = self._load_system_prompt(system_prompt_path)
            self.llm.set_system_prompt(self.system_prompt)
        except Exception as e:
            raise e

    
    @abstractmethod
    def process_message(self, message: str) -> str:
        pass


    @abstractmethod
    def process_message_stream(self, message: str):
        pass

    
    def _load_system_prompt(self, path: str):
        try:
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except FileNotFoundError:
            print(f"Error: The file '{path}' was not found.")
            return None
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            return None