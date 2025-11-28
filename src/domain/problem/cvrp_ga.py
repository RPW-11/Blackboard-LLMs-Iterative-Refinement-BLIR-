import numpy as np
import time
import json
import random
import math
import collections
from typing import List, Dict, Any, Callable, Tuple, Optional
from .problem_base import Problem
from .cvrp_llm_prompt import get_llm_prompt
from .cvrp_type import Instance, Solution
from domain.orchestrator.orchestrator import Orchestrator
from domain.response.solver_agent_response import LargeAgentResponse
from domain.interface.logger import LoggerInterface


class CVRPGeneticAlgorithm(Problem):
    def __init__(
        self,
        description: str,
        orchestrator: Orchestrator,
        logger: LoggerInterface,
        instance: Instance,
        population_size: int = 200,
        max_generations: int = 500,
        tournament_size: int = 5,
        elite_size: int = 10,
        seed: Optional[int] = None
    ):  
        super().__init__(description, orchestrator, logger)
        self.instance = instance
        self.population_size = population_size
        self.max_generations = max_generations
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.random_number_generator = np.random.default_rng(seed)
        self.generation = 0
        self.large_agent_interval = 100 

        self.crossover_operators: List[Tuple[Callable, float]] = [(self._order_crossover, 1.0)]
        self.mutation_operators: List[Tuple[Callable, float]] = [(self._swap_mutation, 1.0)]
        self.repair_operator = self._smart_capacity_repair

        # History and metadata
        self.population_history: List[List[Solution]] = []
        self.best_solution: Optional[Solution] = None
        self.instance_metadata = self._compute_instance_metadata()
        self.historical_data = []

    def solve(self) -> Solution:
        population = self._default_initial_population()
        population = [self._evaluate(chrom) for chrom in population]
        self.population_history.append(population)

        for gen in range(1, self.max_generations + 1):
            self.generation = gen

            # Adaptive large agent call
            if gen % self.large_agent_interval == 0 or gen == 1:
                self._invoke_orchestrator()

            # Selection + Reproduction
            offspring = []
            while len(offspring) < self.population_size:
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                child_chrom = self._crossover(parent1, parent2)
                child_chrom = self._mutate(child_chrom)
                child_chrom = self._repair(child_chrom)
                offspring.append(child_chrom)


            offspring = [self._evaluate(chrom) for chrom in offspring]
            population = self._survival_selection(population + offspring)
            self.population_history.append(population)

            current_best = min(population, key=lambda s: s.total_distance if s.feasible else float('inf'))
            if self.best_solution is None or (current_best.feasible and current_best.total_distance < self.best_solution.total_distance):
                self.best_solution = current_best

            if gen % 50 == 0:
                print(f"Gen {gen} | Best: {self.best_solution.total_distance:.2f} | Feas%: {np.mean([s.feasible for s in population]):.1%}")
                self._save_population_to_memory(population)


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
            feasible = feasible and len(set(chromosome)) == len(self.instance.customers)
        

        return Solution(
            routes=routes,
            total_distance=total_dist,
            feasible=feasible,
            num_vehicles=len(routes),
            metadata={"split_type": "greedy", "chromosome": chromosome.copy()}
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
    
    def _invoke_orchestrator(self):
        context = json.dumps({
            "instance_metadata": self.instance_metadata,
            "current_generation": self.generation,
            "population_stats": self._summarize_population(self.population_history[-1]),
            "historical_patterns": self._summarize_historical_data(),
            "current_operators": [
                {"type": "crossover", "func": op.__name__ if hasattr(op, "__name__") else "lambda", "prob": prob}
                for op, prob in self.crossover_operators
            ] + [
                {"type": "mutation", "func": op.__name__ if hasattr(op, "__name__") else "lambda", "prob": prob}
                for op, prob in self.mutation_operators
            ],
            "best_so_far": self.best_solution.total_distance if self.best_solution else None,
        }, indent=4)

        prompt = get_llm_prompt(self.description, context)

        self.logger.print("="*50 + f"Running the LLM using the following prompt:" + "="*50)
        self.logger.print(prompt)
        self.logger.print("=" * 100)

        result = self.orchestrator.run(prompt, LargeAgentResponse)

        if not result:
            self.logger.print("="*50 + f"There is an LLM error, using default values..." + "="*50)
            self.logger.print("=" * 100)
            return
        
        result = LargeAgentResponse.model_validate(result)

        self._log_result(result)

        self.logger.print("Creating crossover functions....")
        for cross in result.crossover:
            func = self._create_function_safely(cross.code, cross.name)
            if not func:
                self.logger.print("Error creating crossover function!")
                continue

            self.crossover_operators.append((func, cross.prob))
        
        self.logger.print("Creating mutation functions....")
        for mutation in result.mutation:
            func = self._create_function_safely(mutation.code, mutation.name)
            if not func:
                self.logger.print("Error creating mutation function!")
                continue

            self.mutation_operators.append((func, mutation.prob))
        
        if result.repair:
            self.logger.print("Creating repair function....")
            func = self._create_function_safely(result.repair.code, result.repair.name)
            if not func:
                self.logger.print("Error creating repair function!")
            else:
                self.repair_operator = func


    def _create_function_safely(self, func_string, func_name):
        local_scope = {}
        try:
            exec(func_string, globals(), local_scope)
            func = local_scope.get(func_name)
            
            if func and callable(func):
                # Bind the function to the current instance
                bound_func = func.__get__(self, type(self))
                # Replace the method on the instance
                setattr(self, func_name, bound_func)
                return bound_func
            return None
        except Exception as e:
            self.logger.print(f"Error creating function: {str(e)}")
            return None


    def _log_result(self, response: LargeAgentResponse):
        self.logger.print("="*50 + f"Crossover result" + "="*50)
        if len(response.crossover) == 0:
            self.logger.print("No crossover method")
        
        for cross in response.crossover:
            self.logger.print(f"Function Name: {cross.name}")
            self.logger.print(f"Function Prob: {cross.prob}")
            self.logger.print(cross.code)
        
        self.logger.print("="*100)

        # Mutation result
        self.logger.print("="*50 + f"Mutation result" + "="*50)
        if len(response.mutation) == 0:
            self.logger.print("No mutation method")
        
        for mut in response.mutation:
            self.logger.print(f"Function Name: {mut.name}")
            self.logger.print(f"Function Prob: {mut.prob}")
            self.logger.print(mut.code)
        
        self.logger.print("="*100)

        # Repair result
        self.logger.print("="*50 + f"Repair code" + "="*50)
        if response.repair:
            self.logger.print(f"Function Name: {response.repair.name}")
            self.logger.print(response.repair.code)
        else:
            self.logger.print("No repair method")

        self.logger.print("="*100)


    # TOURNAMENT LOGICS
    def _tournament_selection(self, population: List[Solution]) -> List[int]:
        candidates: List[Solution] = self.random_number_generator.choice(population, size=self.tournament_size, replace=False)
        
        feasible_candidates = [s for s in candidates if s.feasible]
        if feasible_candidates:
            winner = min(feasible_candidates, key=lambda s: s.total_distance)
        else:
            winner = min(candidates, key=lambda s: s.total_distance)
        
        return winner.metadata["chromosome"]
    
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
        try:
            if self.random_number_generator.random() < 0.8:
                op, _ = self._sample_operator(self.crossover_operators)
                return op(p1, p2)
            return p1 if self.random_number_generator.random() < 0.5 else p2
        except Exception as e:
            return self._order_crossover(p1, p2)
    
    def _mutate(self, chrom: List[int]) -> List[int]:
        try:
            if self.random_number_generator.random() < 0.2:
                op, _ = self._sample_operator(self.mutation_operators)
                return op(chrom)
            return chrom
        except Exception as e:
            return self._swap_mutation(chrom)

    def _sample_operator(self, operators):
        probs = [p for _, p in operators]
        idx = self.random_number_generator.choice(len(operators), p=np.array(probs)/sum(probs))
        return operators[idx]
    
    def _smart_capacity_repair(self, chromosome: List[int]) -> List[int]:
        if not chromosome:
            return []

        capacity = self.instance.vehicle_capacity
        customers = {c.id: c for c in self.instance.customers}
        
        repaired = []
        current_load = 0
        current_route_customers = []

        for cust_id in chromosome:
            demand = customers[cust_id].demand

            if current_load + demand > capacity and current_route_customers:
                repaired.extend(current_route_customers)
                current_route_customers = []
                current_load = 0

            current_route_customers.append(cust_id)
            current_load += demand

        repaired.extend(current_route_customers)

        if sorted(repaired) != sorted(chromosome):
            repaired = chromosome[:]
            random.shuffle(repaired)

        return repaired

    def _repair(self, chromosome: List[int]) -> List[int]:
        try:
            return self.repair_operator(chromosome)
        except Exception as e:
            self.logger.print(f"Error using LLM's repair function :{str(e)}, falling back to default")
            return self._smart_capacity_repair(chromosome)
    
    
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
    
    def _summarize_population(self, populations: List[Solution]) -> Dict[str, Any]:
        feasible = [s for s in populations if s.feasible]
        return {
            "feasibility_rate": len(feasible) / len(populations),
            "best_distance": min((s.total_distance for s in feasible), default=float('inf')),
            "avg_distance": np.mean([s.total_distance for s in feasible]) if feasible else None,
            "avg_vehicles": np.mean([s.num_vehicles for s in feasible]) if feasible else None,
        }

    def _summarize_historical_data(self) -> List[Dict]:
        self._load_memory()
        return [
            {
                "instance_type": h["instance_type"],
                "best_distance": h["best_distance"],
                "operators_used": h["operators_used"],
            }
            for h in self.historical_data[-50:]
        ]
    
    def _save_population_to_memory(self, population: List[Solution]):
        entry = {
            "timestamp": time.time(),
            "instance_name": self.instance.name,
            "n_customers": self.instance_metadata["n_customers"],
            "instance_type": self.instance_metadata["instance_type"],
            "population_avg_feasibility": np.mean([s.feasible for s in population]),
            "best_distance": min((s.total_distance for s in population if s.feasible), default=float('inf')),
            "n_vehicles": len(population[np.argmin(s.total_distance for s in population if s.feasible)].routes),
            "avg_distance": np.mean([s.total_distance for s in population if s.feasible]),
            "operators_used": [
                {"type": "crossover", "name": op.__name__ if hasattr(op, "__name__") else str(op)}
                for op, _ in self.crossover_operators
            ] + [
                {"type": "mutation", "name": op.__name__ if hasattr(op, "__name__") else str(op)}
                for op, _ in self.mutation_operators
            ],
            "metadata": self.instance_metadata,
        }

        self.orchestrator.save_to_blackboard(entry, file_name=f"{self.generation}_gen_population.json")

    def _load_memory(self):
        self.historical_data = self.orchestrator.load_results(1)
        
    
        

