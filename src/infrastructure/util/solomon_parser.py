from domain.problem.cvrp_type import Instance, Customer
import re
from pathlib import Path


def parse_solomon_instance(file_path: str) -> Instance:
    """
    Parses a Solomon VRPTW benchmark instance (100 customers) from a .txt file.
    
    Args:
        file_path: Path to the Solomon instance file (e.g., C101.txt)
    
    Returns:
        Instance object with depot and list of customers
    """
    path = Path(file_path)
    name = path.stem  # e.g., "C101"

    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # Find vehicle line
    vehicle_line = None
    for line in lines:
        if line.startswith('NUMBER') or 'VEHICLE' in line.upper():
            # Find the line with number and capacity
            for l in lines:
                if re.search(r'\d+\s+\d+', l) and ('25' in l or '200' in l or len(l.split()) == 2):
                    parts = re.split(r'\s+', l.strip())
                    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                        num_vehicles = int(parts[0])
                        capacity = int(parts[1])
                        break
            break

    if 'capacity' not in locals():
        raise ValueError("Could not parse vehicle capacity")

    # Find start of customer data
    customer_start_idx = None
    for i, line in enumerate(lines):
        if 'CUST NO' in line.upper() or ('0   ' in line and '40   50' in line):
            customer_start_idx = i + 1
            break

    if customer_start_idx is None:
        raise ValueError("Could not find customer data section")

    customers = []
    depot = None

    i = customer_start_idx
    while i < len(lines):
        line = lines[i]
        # Skip header-like lines
        if not line[0].isdigit():
            i += 1
            continue

        parts = re.split(r'\s+', line.strip())
        if len(parts) < 7:
            i += 1
            continue

        cust_id = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        demand = int(parts[3])
        ready_time = float(parts[4])
        due_time = float(parts[5])
        service_time = float(parts[6])

        customer = Customer(
            id=cust_id,
            x=x,
            y=y,
            demand=demand,
            ready_time=ready_time,
            due_time=due_time,
            service_time=service_time
        )

        if cust_id == 0:
            depot = customer
        else:
            customers.append(customer)

        i += 1

    if depot is None:
        raise ValueError("Depot (customer 0) not found")

    return Instance(
        name=name,
        depot=depot,
        customers=customers,
        vehicle_capacity=capacity,
        num_vehicles=num_vehicles  # Usually 25 in Solomon, but often ignored in practice
    )