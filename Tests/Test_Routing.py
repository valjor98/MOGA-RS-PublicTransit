import sys
import os

from Routing.Graph import Graph
from Routing.Node import Node
from Routing.Edge import Edge


graph = Graph()

# Add nodes
for i in range(1, 6):
    graph.add_node(Node(i))

# Add edges
graph.add_edge(Edge(1, 2, 2))
graph.add_edge(Edge(1, 3, 4))
graph.add_edge(Edge(2, 3, 1))
graph.add_edge(Edge(2, 4, 7))
graph.add_edge(Edge(3, 4, 3))
graph.add_edge(Edge(3, 5, 5))
graph.add_edge(Edge(4, 5, 7))

print("K-shortest paths:")
k_shortest_paths = graph.yen_k_shortest_paths(1, 5, K=5)
graph.print_paths_with_costs(k_shortest_paths)