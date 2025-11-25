import numpy as np
from typing import List, Tuple, Dict
from math import sqrt
from .cvrp_type import Instance, Customer, Solution
import random


class CVRPRepair:
    def __init__(self, instance: Instance):
        """
        Initialize the repair function for Solomon CVRP.
        
        Args:
            instance: Instance object containing depot, customers, and vehicle capacity
        """
        self.instance = instance
        self.vehicle_capacity = instance.vehicle_capacity
        self.depot_id = instance.depot.id
        
        # Create customer lookup dictionary
        self.customers_dict = {customer.id: customer for customer in instance.customers}
        self.customers_dict[instance.depot.id] = instance.depot
        
        # Get all customer IDs (excluding depot)
        self.customer_ids = [c.id for c in instance.customers if c.id != self.depot_id]
        self.n_customers = len(self.customer_ids)
        
    def repair_chromosome(self, chromosome: List[int]) -> List[int]:
        """
        Repair a chromosome to ensure it represents valid CVRP routes.
        Returns a chromosome of exactly 100 customer indices that when split 
        using greedy capacity method will produce feasible routes.
        """
        # Step 1: Fix missing/duplicate customers and ensure correct length
        chromosome = self._fix_missing_duplicate_customers(chromosome)
        
        # Step 2: Reorder chromosome to ensure capacity feasibility
        chromosome = self._reorder_for_feasibility(chromosome)
        
        return chromosome
    
    def _fix_missing_duplicate_customers(self, chromosome: List[int]) -> List[int]:
        """Ensure all 100 customers appear exactly once and no depot."""
        # Remove depot and invalid customer IDs
        valid_chromosome = []
        for gene in chromosome:
            if gene != self.depot_id and gene in self.customer_ids:
                valid_chromosome.append(gene)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_chromosome = []
        for gene in valid_chromosome:
            if gene not in seen:
                seen.add(gene)
                unique_chromosome.append(gene)
        
        # Add missing customers
        missing_customers = list(set(self.customer_ids) - seen)
        
        # Insert missing customers using best-insertion heuristic
        for customer in missing_customers:
            best_position = self._find_best_insertion_position(unique_chromosome, customer)
            unique_chromosome.insert(best_position, customer)
        
        # Ensure correct length
        if len(unique_chromosome) > self.n_customers:
            unique_chromosome = unique_chromosome[:self.n_customers]
        
        return unique_chromosome
    
    def _find_best_insertion_position(self, chromosome: List[int], customer: int) -> int:
        """Find the best position to insert a customer to maintain feasibility."""
        customer_demand = self.customers_dict[customer].demand
        
        # Try to find a position where insertion won't break capacity constraints
        best_position = 0
        best_feasibility_score = float('inf')
        
        for i in range(len(chromosome) + 1):
            # Test insertion at position i
            test_chromosome = chromosome[:i] + [customer] + chromosome[i:]
            
            # Check feasibility of this insertion
            feasibility_score = self._calculate_feasibility_score(test_chromosome)
            
            if feasibility_score == 0:  # Perfectly feasible
                return i
            elif feasibility_score < best_feasibility_score:
                best_feasibility_score = feasibility_score
                best_position = i
        
        return best_position
    
    def _reorder_for_feasibility(self, chromosome: List[int]) -> List[int]:
        """
        Reorder chromosome to ensure capacity feasibility when split greedily.
        Uses a combination of largest-demand-first and spatial clustering.
        """
        # Create a copy to work with
        current_chromosome = chromosome.copy()
        
        # Try multiple reordering strategies
        strategies = [
            self._reorder_by_demand_clustering,
            self._reorder_by_savings_heuristic,
            self._reorder_by_route_building
        ]
        
        best_chromosome = current_chromosome
        best_feasibility_score = self._calculate_feasibility_score(current_chromosome)
        
        # If already feasible, return as is
        if best_feasibility_score == 0:
            return current_chromosome
        
        # Try each strategy
        for strategy in strategies:
            reordered = strategy(current_chromosome)
            score = self._calculate_feasibility_score(reordered)
            
            if score == 0:  # Found feasible ordering
                return reordered
            elif score < best_feasibility_score:
                best_feasibility_score = score
                best_chromosome = reordered
        
        # If still not feasible, use iterative improvement
        improved_chromosome = self._iterative_improvement(best_chromosome)
        return improved_chromosome
    
    def _reorder_by_demand_clustering(self, chromosome: List[int]) -> List[int]:
        """Reorder by grouping high-demand customers separately."""
        # Sort customers by demand (descending)
        customers_sorted = sorted(chromosome, 
                                key=lambda c: self.customers_dict[c].demand, 
                                reverse=True)
        
        # Interleave high and low demand customers to balance routes
        reordered = []
        n = len(customers_sorted)
        half = n // 2
        
        for i in range(half):
            reordered.append(customers_sorted[i])  # High demand
            if i + half < n:
                reordered.append(customers_sorted[i + half])  # Low demand
        
        # Add remaining customers if odd number
        if n % 2 == 1:
            reordered.append(customers_sorted[-1])
            
        return reordered
    
    def _reorder_by_route_building(self, chromosome: List[int]) -> List[int]:
        """Build routes sequentially to ensure capacity feasibility."""
        remaining_customers = set(chromosome)
        reordered = []
        current_load = 0
        
        while remaining_customers:
            # Find customers that can be added without exceeding capacity
            feasible_next = []
            for customer in remaining_customers:
                demand = self.customers_dict[customer].demand
                if current_load + demand <= self.vehicle_capacity:
                    feasible_next.append(customer)
            
            if feasible_next:
                # Choose the customer with highest demand first
                next_customer = max(feasible_next, 
                                  key=lambda c: self.customers_dict[c].demand)
                reordered.append(next_customer)
                current_load += self.customers_dict[next_customer].demand
                remaining_customers.remove(next_customer)
            else:
                # Start new route
                current_load = 0
                # Add the smallest customer that fits
                smallest_customer = min(remaining_customers, 
                                      key=lambda c: self.customers_dict[c].demand)
                reordered.append(smallest_customer)
                current_load += self.customers_dict[smallest_customer].demand
                remaining_customers.remove(smallest_customer)
        
        return reordered
    
    def _reorder_by_savings_heuristic(self, chromosome: List[int]) -> List[int]:
        """Use savings heuristic to create a good customer sequence."""
        # This is a simplified version of Clark & Wright savings
        customers = chromosome.copy()
        random.shuffle(customers)  # Start with random order
        
        # Try to improve by swapping adjacent customers
        improved = True
        while improved:
            improved = False
            for i in range(len(customers) - 1):
                # Try swapping adjacent customers
                new_chromosome = customers.copy()
                new_chromosome[i], new_chromosome[i + 1] = new_chromosome[i + 1], new_chromosome[i]
                
                if self._calculate_feasibility_score(new_chromosome) < self._calculate_feasibility_score(customers):
                    customers = new_chromosome
                    improved = True
                    break
        
        return customers
    
    def _iterative_improvement(self, chromosome: List[int]) -> List[int]:
        """Iteratively improve chromosome feasibility."""
        current = chromosome.copy()
        current_score = self._calculate_feasibility_score(current)
        
        for _ in range(100):  # Maximum iterations
            # Generate neighbor by swapping two random customers
            i, j = random.sample(range(len(current)), 2)
            neighbor = current.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            
            neighbor_score = self._calculate_feasibility_score(neighbor)
            
            if neighbor_score < current_score:
                current = neighbor
                current_score = neighbor_score
            
            if current_score == 0:  # Found feasible solution
                break
        
        return current
    
    def _calculate_feasibility_score(self, chromosome: List[int]) -> float:
        """
        Calculate a feasibility score for the chromosome.
        Lower score is better, 0 means perfectly feasible.
        """
        if not chromosome:
            return float('inf')
        
        # Simulate the greedy splitting process
        capacity = self.vehicle_capacity
        current_load = 0
        overload_penalty = 0
        route_count = 0
        
        for cust_id in chromosome:
            demand = self.customers_dict[cust_id].demand
            
            if current_load + demand > capacity:
                if current_load > 0:
                    route_count += 1
                # Calculate how much we're over capacity
                overload = max(0, current_load + demand - capacity)
                overload_penalty += overload * 1000  # Heavy penalty for overloads
                current_load = demand
            else:
                current_load += demand
        
        if current_load > 0:
            route_count += 1
        
        # Also check if all customers are unique
        uniqueness_penalty = 0
        if len(set(chromosome)) != len(chromosome):
            uniqueness_penalty = 10000
        
        return overload_penalty + uniqueness_penalty + route_count
    
    def validate_chromosome_feasibility(self, chromosome: List[int]) -> Tuple[bool, str]:
        """
        Validate if the chromosome will produce feasible routes when split greedily.
        """
        if len(chromosome) != self.n_customers:
            return False, f"Wrong chromosome length: {len(chromosome)} != {self.n_customers}"
        
        if len(set(chromosome)) != len(chromosome):
            return False, "Duplicate customers found"
        
        # Simulate the greedy splitting to check feasibility
        capacity = self.vehicle_capacity
        current_load = 0
        route_overloads = []
        
        for cust_id in chromosome:
            demand = self.customers_dict[cust_id].demand
            
            if current_load + demand > capacity:
                if current_load > 0:
                    # Check if previous route was feasible
                    if current_load > capacity:
                        route_overloads.append(current_load)
                current_load = demand
            else:
                current_load += demand
        
        # Check last route
        if current_load > capacity:
            route_overloads.append(current_load)
        
        if route_overloads:
            return False, f"Routes exceed capacity: {route_overloads}"
        
        return True, "Chromosome produces feasible routes"