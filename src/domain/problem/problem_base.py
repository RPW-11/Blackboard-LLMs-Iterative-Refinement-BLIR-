from abc import ABC, abstractmethod
from typing import Any
from domain.orchestrator.orchestrator import Orchestrator
from domain.interface.logger import LoggerInterface


class Problem(ABC):
    def __init__(self, description: str, orchestrator: Orchestrator, logger: LoggerInterface):
        self.description = description
        self.orchestrator = orchestrator
        self.logger = logger

    @abstractmethod
    def solve(self) -> Any:
        pass