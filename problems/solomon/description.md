The **Solomon 100 CVRP** is not a single problem, but a famous and widely used set of **100 benchmark instances** for the **Capacitated Vehicle Routing Problem (CVRP)**.

Here are its key characteristics:

*   **Core Problem:** It is a classic CVRP where the goal is to find the optimal set of routes for a fleet of identical vehicles to serve a set of geographically dispersed customers from a single central depot. Each vehicle has a fixed capacity, and each customer has a specific demand.

*   **What Makes it Special:** The dataset is designed to test algorithms on different spatial distributions of customers, which are categorized into six types:
    1.  **C1, C2:** *Clustered* customers.
    2.  **R1, R2:** *Randomly* distributed customers.
    3.  **RC1, RC2:** A *mix* of Random and Clustered customers.

*   **The "100" in the Name:** The "100" refers to the number of customers to be served in each instance (plus one central depot).

*   **Key Challenge:** The primary challenge lies in the "capacity" constraint. The problems are structured so that for the "1" series (R1, C1, RC1), the vehicle capacity is relatively small, requiring many short routes. For the "2" series (R2, C2, RC2), the vehicle capacity is larger, allowing for fewer but longer routes. This tests an algorithm's ability to handle different route structures and consolidation strategies.