def get_llm_prompt(description: str, context: str|dict):
    prompt = f"""
You are trying to solve a CVRP problem as described in the following:
{description}

The following is the current information of the solving process:
Context:
{context}

The following are variables that you can use to write down your functions (must be accessed with 'self'):
self.instance
self.population_size = population_size
self.max_generations = max_generations
self.tournament_size = tournament_size
self.elite_size = elite_size

These are some types used in the class:
class Customer:
    id: int # ID 0 means the depot
    x: float
    y: float
    demand: int
    ready_time: float = 0.0
    due_time: float = 0.0
    service_time: float = 0.0

class Instance:
    name: str
    depot: Customer
    customers: List[Customer]
    vehicle_capacity: int
    num_vehicles: int = 1000

class Solution:
    routes: List[List[int]]
    total_distance: float = 0.0
    feasible: bool = False
    num_vehicles: int = 0
    metadata: Dict[str, Any] = None

Imported libraries (have been imported):
import numpy as np
import time
import json
import random
import math
import collections
from typing import List, Dict, Any, Callable, Tuple, Optional
from dataclasses import dataclass, asdict

For writing the crossover operators, the arguments must only be parent 1 and parent 2, both having a type of List[int].
The same way for writing mutation operators, where the input is a chromosome with a type of List[int].
REMEMBER, THERE IS NO chromosome with a value of 0 because it is the depot!
Below are some examples:
1. Crossover function:
def crossover_a(self, parent1: List[int], parent2: List[int]) -> List[int]:
    ....

2. Mutation function:
def mutation_a(self, chromosome: List[int]) -> List[int]:
    ...
"""
    return prompt.strip()