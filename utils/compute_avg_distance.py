import os
import math
import csv

def compute_average_distances(directory_path, output_csv):
    """
    Reads all Solomon-100 instance files in the given directory, computes the average Euclidean distance
    between all pairs of cities (including the depot) for each instance, and stores the results in a CSV file.
    
    :param directory_path: Path to the directory containing the .txt instance files.
    :param output_csv: Path to the output CSV file.
    """
    results = []
    
    for filename in os.listdir(directory_path):
        if not filename.lower().endswith('.txt'):
            continue
            
        instance_name = os.path.splitext(filename)[0]
        file_path = os.path.join(directory_path, filename)
        
        points = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        parsing_customers = False
        for line in lines:
            line = line.strip()
            if not line:
                continue  # Skip empty lines but keep parsing
            
            if line == 'CUSTOMER':
                parsing_customers = True
                continue
            
            if not parsing_customers:
                continue
            
            # Skip the header line
            if line.startswith('CUST NO.'):
                continue
            
            # Split into at most 8 parts to handle variable whitespace
            parts = line.split(None, 8)
            
            if len(parts) < 7:
                continue  # Not enough columns
            
            # Customer number should be a digit
            if not parts[0].isdigit():
                continue
            
            try:
                cust_no = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                points.append((x, y))
            except ValueError:
                # Skip any malformed line
                continue
        
        if len(points) < 2:
            print(f"No valid points found in {filename} (found {len(points)} lines)")
            continue
        
        # Compute average Euclidean distance over all pairs
        n = len(points)
        total_dist = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                dist = math.sqrt(dx**2 + dy**2)
                total_dist += dist
                count += 1
        
        if count > 0:
            average_dist = total_dist / count
            results.append((instance_name, average_dist))
            print(f"{instance_name}: {len(points)} points, avg dist = {average_dist:.4f}")
        else:
            print(f"No pairs computed for {filename}")
    
    # Sort results by instance name for consistency (optional)
    results.sort(key=lambda x: x[0])
    
    # Write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Instance', 'Average Distance'])
        for row in results:
            writer.writerow([row[0], f"{row[1]:.4f}"])
    
    print(f"\nResults saved to {output_csv} ({len(results)} instances processed)")

# Example usage:
compute_average_distances('/Users/rainataputra/Projects/MultiAgentSolver/problems/solomon/dataset', 'solomon_average_distances.csv')