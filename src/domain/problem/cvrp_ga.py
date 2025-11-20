import numpy as np
from typing import List, Dict, Any, Callable, Tuple, Optional
from dataclasses import dataclass, asdict
from .problem_base import Problem
from domain.orchestrator.orchestrator import Orchestrator
from domain.response.solver_agent_response import LargeAgentResponse


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


class CVRPGeneticAlgorithm(Problem):
    def __init__(
        self,
        description: str,
        orchestrator: Orchestrator,
        instance: Instance,
        population_size: int = 200,
        max_generations: int = 500,
        tournament_size: int = 5,
        elite_size: int = 10,
        seed: Optional[int] = None
    ):  
        super().__init__(description, orchestrator)
        self.instance = instance
        self.population_size = population_size
        self.max_generations = max_generations
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.random_number_generator = np.random.default_rng(seed)
        self.generation = 0

        # LLM-generated components (below are just the default settings)
        self.initial_population_generator = self._default_initial_population
        self.crossover_operators: List[Tuple[Callable, float]] = [(self._order_crossover, 1.0)]
        self.mutation_operators: List[Tuple[Callable, float]] = [(self._swap_mutation, 1.0)]
        self.repair_operator = self._basic_capacity_repair
        self.local_search_operator = self._noop_local_search

        # History and metadata
        self.population_history: List[List[Solution]] = []
        self.best_solution: Optional[Solution] = None
        self.instance_metadata = self._compute_instance_metadata()

    def solve(self) -> Solution:
        population = self.initial_population_generator()
        population = [self._evaluate(chrom) for chrom in population]
        self.population_history.append(population)

        for gen in range(1, self.max_generations + 1):
            self.generation = gen

            # Adaptive large agent call
            # if gen % self.large_agent_interval == 0 or gen == 1:
            #     self._invoke_large_research_agent()

            # Selection + Reproduction
            offspring = []
            while len(offspring) < self.population_size:
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                child_chrom = self._crossover(parent1, parent2)
                # child_chrom = self._mutate(child_chrom)
                # child_chrom = self.repair_operator(child_chrom)
                # child_chrom = self.local_search_operator(child_chrom)
                if (not any(c == -1 for c in child_chrom)):
                    offspring.append(child_chrom)
                # offspring.append(child_chrom)


            offspring = [self._evaluate(chrom) for chrom in offspring]
            population = self._survival_selection(population + offspring)
            self.population_history.append(population)

            current_best = min(population, key=lambda s: s.total_distance if s.feasible else float('inf'))
            if self.best_solution is None or (current_best.feasible and current_best.total_distance < self.best_solution.total_distance):
                self.best_solution = current_best

            if gen % 50 == 0:
                print(f"Gen {gen} | Best: {self.best_solution.total_distance:.2f} | Feas%: {np.mean([s.feasible for s in population]):.1%}")

        
        self.orchestrator.save_to_blackboard({"population": [p.to_dict() for p in population]})
        self.orchestrator.run("Your are solving a GA problem about CVRP Salomon, please return answers based on the given structure", LargeAgentResponse)

        return self.best_solution

    def _default_initial_population(self) -> List[List[int]]:
        customers = [c.id for c in self.instance.customers if c.id != 0]
        population = []
        for _ in range (self.population_size):
            chrom = customers.copy()
            self.random_number_generator.shuffle(chrom)
            population.append(chrom)
        return population
    
    def _evaluate(self, chromosome: List[int]) -> Solution:
        """Split chromosome into routes using capacity, compute distance."""
        capacity = self.instance.vehicle_capacity
        depot = self.instance.depot
        customers = {c.id: c for c in self.instance.customers}
        current_load = 0
        routes = []
        current_route = []

        for cust_id in chromosome:
            demand = customers[cust_id].demand
            if current_load + demand > capacity and current_route:
                routes.append(current_route)
                current_route = []
                current_load = 0
            current_route.append(cust_id)
            current_load += demand

        if current_route:
            routes.append(current_route)

        total_dist = 0.0
        for route in routes:
            dist = self._route_distance([0] + route + [0])
            total_dist += dist

        feasible = all(sum(customers[c].demand for c in route) <= capacity for route in routes)
        if chromosome:
            feasible = feasible and len(set(chromosome)) == len(self.instance.customers) - 1

        return Solution(
            routes=routes,
            total_distance=total_dist,
            feasible=feasible,
            num_vehicles=len(routes),
            metadata={"split_type": "greedy"}
        )
    
    def _route_distance(self, route: List[int]) -> float:
        """Sum of the euclidean distance from two consecutive customers"""
        customers = {c.id: c for c in [self.instance.depot] + self.instance.customers}
        dist = 0.0
        for i in range(len(route) - 1):
            a = customers[route[i]]
            b = customers[route[i + 1]]
            dist += np.hypot(a.x - b.x, a.y - b.y)
        return dist
    
    # TOURNAMENT LOGICS
    def _tournament_selection(self, population: List[Solution]) -> List[int]:
        candidates: List[Solution] = self.random_number_generator.choice(population, size=self.tournament_size, replace=False)
        valid = [s for s in candidates if s.feasible]
        if valid:
            return min(valid, key=lambda s:s.total_distance).routes[0]
        return min(candidates, key=lambda s: s.total_distance).routes[0]
    
    def _survival_selection(self, combined: List[Solution]) -> List[Solution]:
        combined = sorted(combined, key=lambda s: (not s.feasible, s.total_distance))
        return combined[:self.population_size]
    
    # CROSSOVER AND MUTATION LOGICS
    def _order_crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        size = len(p1)
        a, b = sorted(self.random_number_generator.integers(0, size, 2))
        child = [-1] * size
        child[a:b] = p1[a:b]
        remaining = [x for x in p2 if x not in child[a:b]]
        child[:a] = remaining[:a]
        child[b:] = remaining[a:]
        return child
    
    def _swap_mutation(self, chrom: List[int]) -> List[int]:
        if len(chrom) < 2:
            return chrom
        
        i, j = self.random_number_generator.integers(0, len(chrom), 2)
        chrom[i], chrom[j] = chrom[j], chrom[i]
        return chrom
    
    def _crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        if self.random_number_generator.random() < 0.8:
            op, _ = self._sample_operator(self.crossover_operators)
            return op(p1, p2)
        return p1 if self.random_number_generator.random() < 0.5 else p2
    
    def _mutate(self, chrom: List[int]) -> List[int]:
        if self.random_number_generator.random() < 0.2:
            op, _ = self._sample_operator(self.mutation_operators)
            return op(chrom)
        return chrom

    def _sample_operator(self, operators):
        probs = [p for _, p in operators]
        idx = self.random_number_generator.choice(len(operators), p=np.array(probs)/sum(probs))
        return operators[idx]
    
    #
    def _basic_capacity_repair(self, chromosome: List[int]) -> List[int]:
        """Extract overloaded customers and reinsert with cheapest insertion."""
        pass  # LLM will replace this

    def _noop_local_search(self, chromosome: List[int]) -> List[int]:
        return chromosome
    
    def _compute_instance_metadata(self) -> Dict[str, Any]:
        cust = [c for c in self.instance.customers if c.id != 0]
        demands = [c.demand for c in cust]
        coords = np.array([(c.x, c.y) for c in cust])
        dist_matrix = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(-1))
        avg_dist = dist_matrix.mean()
        capacity = self.instance.vehicle_capacity
        total_demand = sum(demands)
        avg_demand = np.mean(demands)
        density = len(cust) / (np.ptp(coords[:, 0]) * np.ptp(coords[:, 1]) + 1e-8)

        return {
            "n_customers": len(cust),
            "total_demand": total_demand,
            "avg_demand": avg_demand,
            "capacity": capacity,
            "load_factor": total_demand / (capacity * 10),  # assume ~10 vehicles typical
            "avg_pairwise_dist": avg_dist,
            "spatial_density": density,
            "instance_type": self._guess_instance_type(coords),
        }
    
    def _guess_instance_type(self, coords: np.ndarray) -> str:
        from sklearn.cluster import KMeans
        if len(coords) < 10:
            return "RC"
        kmeans = KMeans(n_clusters=3, n_init=10).fit(coords)
        cluster_sizes = np.bincount(kmeans.labels_)

        if max(cluster_sizes) > 0.7 * len(coords):
            return "C" # clustered
        return "R" # random
    
    def get_best_result(self) -> Solution:
        return self.best_solution
    
    def apply_llm_response(self, response):
        return super().apply_llm_response(response)
        

