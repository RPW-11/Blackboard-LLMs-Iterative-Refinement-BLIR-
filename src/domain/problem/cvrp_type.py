from dataclasses import dataclass, asdict
from typing import List, Dict, Any



@dataclass
class Customer:
    id: int
    x: float
    y: float
    demand: int
    ready_time: float = 0.0
    due_time: float = 0.0
    service_time: float = 0.0

@dataclass
class Instance:
    name: str
    depot: Customer
    customers: List[Customer]
    vehicle_capacity: int
    num_vehicles: int = 1000

@dataclass
class Solution:
    routes: List[List[int]]
    total_distance: float = 0.0
    feasible: bool = False
    num_vehicles: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> str:
        return asdict(self)