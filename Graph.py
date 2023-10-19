from heapq import heappop, heappush

class Graph:
    def __init__(self):
        self.nodes = {}  # Node ID -> Node object
        self.edges = []  # List of Edge objects

    def add_node(self, node):
        self.nodes[node.node_id] = node

    def add_edge(self, edge):
        self.edges.append(edge)

    def dijkstra(self, start_node_id, end_node_id):
        distances = {node: float('inf') for node in self.nodes}
        distances[start_node_id] = 0
        priority_queue = [(0, start_node_id)]
        
        while priority_queue:
            current_distance, current_node_id = heappop(priority_queue)
            
            if current_distance > distances[current_node_id]:
                continue

            for edge in self.edges:
                if edge.start_node == current_node_id:
                    distance = current_distance + edge.weight
                    if distance < distances[edge.end_node]:
                        distances[edge.end_node] = distance
                        heappush(priority_queue, (distance, edge.end_node))

        return distances[end_node_id]
