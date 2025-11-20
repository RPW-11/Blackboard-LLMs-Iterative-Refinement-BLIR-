from abc import ABC, abstractmethod
from typing import Any
from domain.orchestrator.orchestrator import Orchestrator


class Problem(ABC):
    def __init__(self, description: str, orchestrator: Orchestrator):
        self.description = description
        self.orchestrator = orchestrator

    @abstractmethod
    def get_best_result(self)->dict:
        pass

    @abstractmethod
    def solve(self) -> Any:
        pass