from abc import ABC, abstractmethod
from typing import List


class LoggerInterface(ABC):
    @abstractmethod
    def print(self, msg: str):
        pass

    