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

    def yen_k_shortest_paths(self, start_node_id, end_node_id, K=3):
        A = []  # Holds the k shortest paths
        B = []  # Candidate paths
        
        # 1. Find the shortest path from start to end
        initial_path = self.dijkstra(start_node_id, end_node_id)
        A.append(initial_path)
        
        # 3. Main loop for k = 1 to K
        for k in range(1, K):
            # 3.1 For each node n in the k-1th shortest path
            for i in range(len(A[-1]) - 1):
                spur_node = A[-1][i]
                root_path = A[-1][:i]
                
                # 3.1.3 Remove all nodes in root path from the graph
                self.remove_nodes(root_path)
                
                # 3.1.4 Find the spur path
                spur_path = self.dijkstra(spur_node, end_node_id)
                
                # 3.1.6 Concatenate root and spur paths
                total_path = root_path + spur_path
                
                # 3.1.7 Add the new path to B
                if total_path not in A and total_path not in B:
                    heappush(B, total_path)
            
            # 3.2 Add the shortest path in B to A
            if B:
                shortest_path_B = heappop(B)
                A.append(shortest_path_B)
                
        return A