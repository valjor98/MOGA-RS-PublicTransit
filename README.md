# A Multi-Objective Genetic Algorithm Approach for Co-Optimizing Routing and Scheduling in Public Transit

## Overview

This project introduces a novel methodology using Multi-Objective Genetic Algorithms (MOGA) to enhance the efficiency and effectiveness of public transit systems. The methodology integrates route optimization and vehicle scheduling to balance operational costs and service quality. The approach was validated using real data from public buses in Monterrey, Mexico, aiming to create a more sustainable and efficient public transportation framework.

## Abstract

Public transportation efficiency in designing routes and scheduling vehicles is a complex, multi-dimensional problem. This project presents a methodology based on Multi-Objective Evolutionary Algorithms to handle conflicting objectives effectively. The study addresses the design of optimal routes followed by vehicle scheduling, considering operational costs and unserved start times. The results, validated using real data from Monterrey, aim to improve the public transportation system, enhancing the quality of life for its residents.

## Key Features

- **Route Optimization:** Utilizes Yen’s K Shortest Paths algorithm to generate multiple efficient paths considering various attributes such as priority levels and zone types.
- **Scheduling Optimization:** Employs the Non-dominated Sorting Genetic Algorithm II (NSGA-II) to optimize scheduling, focusing on minimizing operational costs and improving service efficiency.
- **Multi-Objective Approach:** Balances different criteria including operational costs, service regularity, and coverage, providing a holistic optimization solution.
- **Real-World Validation:** Validated with real public transit data from Monterrey, demonstrating significant improvements in route efficiency and scheduling effectiveness.

## Methodology

### 1. Routing

- **Graph Representation:** The transportation network is represented as a weighted graph where nodes correspond to transit points and edges signify possible routes between nodes.
- **Initialization:** Sets up the graph and prepares for the execution of Yen’s K Shortest Paths algorithm.
- **Pathfinding Algorithms:** Uses Dijkstra’s algorithm within Yen’s K Shortest Paths to find multiple optimal paths.
- **Path Selection and Cost Calculation:** Ensures the integrity of the graph and calculates the total cost for each path based on edge weights.

### 2. Scheduling

- **Individual Representation:** Each solution is represented by a set of vehicle schedules and routes, evaluated based on uncovered start times, number of vehicles, and route cost.
- **Population Initialization:** Creates a diverse set of initial solutions using randomly generated vehicle blocks and precomputed routes.
- **Objective Function Evaluation:** Evaluates solutions based on minimizing uncovered start times, number of blocks, and route costs.
- **Evolutionary Process:** Uses selection, crossover, and mutation mechanisms to refine the population towards optimal solutions over generations.

## Experiment Design

The experiment was conducted using a modeled graph of a university area in Monterrey, consisting of 23 nodes representing significant locations. Two schedules were optimized: one from Node 1 to Node 23 and another from Node 23 to Node 1. The optimization aimed to find Pareto optimal solutions focusing on minimizing unserved start times, number of buses, and route costs.

## Results

- **Schedule from Node 1 to Node 23:** The optimized schedule matched the current route distance of 1800 meters while improving other aspects like the number of buses and unserved start times.
- **Schedule from Node 23 to Node 1:** The optimized route showed a reduction in total distance compared to the existing route, with improved efficiency in bus usage and scheduling.

## Conclusions

The application of a multi-objective genetic algorithm successfully optimized the routing and scheduling of public transit in Monterrey. The methodology demonstrated significant improvements in operational efficiency and service quality, offering a robust tool for urban transit planning. The approach aligns with sustainability goals by optimizing resource utilization and reducing environmental impact.

## Technical Specifications

- **Development Environment:** Python 3.8
- **System Specifications:** Intel Core i3-10110U processor, 8GB RAM, Windows 11-64 bit

## References

1. Mohammad Hadi Almasi, et al., "Urban transit network optimization under variable demand with single and multi-objective approaches using metaheuristics," International Journal of Sustainable Transportation, 2020. DOI: [10.1080/15568318.2020.1821414](https://doi.org/10.1080/15568318.2020.1821414)
2. K. Deb, et al., "A fast and elitist multiobjective genetic algorithm: NSGA-II," IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, pp. 182–197, 2002.
3. Edsger W. Dijkstra, "A note on two problems in connexion with graphs," Numerische Mathematik, vol. 1, no. 1, pp. 269–271, 1959.
4. H. Knut, et al., "Simultaneous vehicle and crew scheduling in urban mass transit systems," Transportation Science, vol. 35, no. 3, pp. 286–303, 2001.
5. Y. Lin, et al., "A bi-level multi-objective programming model for bus crew and vehicle scheduling," In 2010 International Conference on Intelligent Control and Automation (WCICA), IEEE, 2010, pp. 2328–2333.
6. Chunlu Wang, et al., "A multi-objective genetic algorithm based approach for dynamical bus vehicles scheduling under traffic congestion," Swarm and Evolutionary Computation, vol. 54, 2020. DOI: [10.1016/j.swevo.2020.100667](https://doi.org/10.1016/j.swevo.2020.100667)
7. Jin Y. Yen, "An algorithm for finding shortest routes from all source nodes to a given destination in general networks," Quart. Appl. Math., vol. 27, pp. 526–530, 1970. DOI: [10.1090/qam/253822](https://doi.org/10.1090/qam/253822)
