import sys
import os

# Add the project's root directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root)


from Routing.Graph import Graph
from Routing.Node import Node
from Routing.Edge import Edge

from Scheduling.Scheduling_NSGA_II import NSGA_II, plot_pareto_front

import csv

def initialize_graph_from_csv(nodes_csv, edges_csv):
    graph = Graph()

    # Read and add nodes
    with open(nodes_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            node = Node(int(row['id']), priority=int(row['priority']), zone_type=int(row['zone_type']))
            graph.add_node(node)

    # Read and add edges
    with open(edges_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            edge = Edge(int(row['start_node']), int(row['end_node']), int(row['weight']))
            graph.add_edge(edge)

    return graph

def generate_schedule(start_hour, end_hour, intervals):
    """
    Generate a schedule in minutes after midnight given start and end hours, and intervals.
    
    Parameters:
        start_hour: The hour at which the schedule starts (e.g., 6 for 6:00 AM).
        end_hour: The hour at which the schedule ends (e.g., 19 for 7:15 PM).
        intervals: List of minute intervals past each hour when a bus arrives.
    
    Returns:
        A list of times in minutes after midnight.
    """
    schedule = []
    for hour in range(start_hour, end_hour + 1):
        for interval in intervals:
            time = hour * 60 + interval
            if time <= end_hour * 60 + 15:  # Ensuring we don't go beyond 19:15
                schedule.append(time)
    return schedule

def main():
    graph = initialize_graph_from_csv('graph_Nodes.csv', 'graph_Edges.csv')

    # Running it fron Node 1 to Node 23
    k_shortest_paths_1 = graph.yen_k_shortest_paths(1, 23, 10)
    path_costs_1 = [graph.path_cost(path) for path in k_shortest_paths_1]
    print("First half of the trip")
    graph.print_paths_with_costs(k_shortest_paths_1)
    schedule_1 = generate_schedule(6, 19, [0, 15, 30, 45])
    print("Schedule 1:", schedule_1)


    # NSGA-II parameters
    pop_size = 100
    generations = 120
    crossover_prob = 0.9
    mutation_prob = 0.1
    max_blocks_per_individual = 12
    max_block_length = 8
    target_schedule = schedule_1
    k_shortest_paths = k_shortest_paths_1
    node_attributes=graph.nodes
    #path_costs=path_costs_1

    
    # Run NSGA-II algorithm with K shortest paths
    population, best_solution = NSGA_II(
        pop_size, 
        generations, 
        crossover_prob, 
        mutation_prob, 
        max_blocks_per_individual, 
        max_block_length, 
        target_schedule, 
        k_shortest_paths,
        node_attributes,
        #path_costs
    )
    print("Best Individual Details:")
    print("Blocks:", best_solution.blocks)
    print("Fitness:", best_solution.fitness)
    print("Route:", best_solution.route)

    plot_pareto_front(population)

    """
    # Running it from Node 23 to Node 1
    k_shortest_paths_2 = graph.yen_k_shortest_paths(23, 1, 10)
    path_costs_2 = [graph.path_cost(path) for path in k_shortest_paths_2]
    print("Second half of the trip")
    graph.print_paths_with_costs(k_shortest_paths_2)
    schedule_2 = generate_schedule(6, 19, [11, 26, 41, 56])
    print("Schedule 2:", schedule_2)

    target_schedule = schedule_2
    k_shortest_paths = k_shortest_paths_2

    # Run NSGA-II algorithm with K shortest paths
    population, best_solution = NSGA_II(
        pop_size, 
        generations, 
        crossover_prob, 
        mutation_prob, 
        max_blocks_per_individual, 
        max_block_length, 
        target_schedule, 
        k_shortest_paths,
        node_attributes,
        #path_costs
    )
    print("Best Individual Details:")
    print("Blocks:", best_solution.blocks)
    print("Fitness:", best_solution.fitness)
    print("Route:", best_solution.route)

    plot_pareto_front(population)
    """
    

if __name__ == "__main__":
    main()