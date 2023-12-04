from heapq import heappop, heappush
from Routing.Edge import Edge

class Graph:
    def __init__(self):
        self.nodes = {}  # Node ID -> Node object
        self.edges = []  # List of Edge objects
        self.removed_nodes = set()
        self.removed_edges = set()


    def add_node(self, node):
        self.nodes[node.id] = node

    def add_edge(self, edge):
        self.edges.append(edge)

    def remove_nodes(self, node_ids):
        self.removed_nodes = set(node_ids)

    def remove_edge(self, edge):
        self.removed_edges.add((edge.start_node, edge.end_node))

    def restore_nodes(self):
        self.removed_nodes = set()

    def restore_edges(self):
        self.removed_edges = set()


    # O(nlogn + mlogn)
    def dijkstra(self, start_node_id, end_node_id):
        # Starts by setting the best distances of each node to infinity
        distances = {node: float('inf') for node in self.nodes}

        previous_nodes = {node: None for node in self.nodes}
        # Sets the distance of the start node to 0
        distances[start_node_id] = 0
        # Initialization of the priority queue
        priority_queue = [(0, start_node_id)]

        # As long as priority queue is not empty
        while priority_queue:
            # The node with the smallest distance is popped from the priority queue
            current_distance, current_node_id = heappop(priority_queue)
            # if the current's node distance is greater than the smallest known distance to this noce, we skip it
            if current_distance > distances[current_node_id]:
                continue


            # for each edge that starts with the current node
            for edge in self.edges:
                # if the start node of that particular edge is our current node
                if edge.start_node == current_node_id and edge.end_node not in self.removed_nodes and (edge.start_node, edge.end_node) not in self.removed_edges:
                    # sum the smallest known distance to the current node and the weight of the edge
                    distance = current_distance + edge.weight
                    # check if the new distance is smaller than the currently known shortest distance to the end node of the edge
                    if distance < distances[edge.end_node]:
                        # update shortest known distance
                        distances[edge.end_node] = distance
                        previous_nodes[edge.end_node] = current_node_id
                        # push to priority queue
                        heappush(priority_queue, (distance, edge.end_node))

        # Construct the shortest path
        path = []
        current = end_node_id
        while current is not None:
            path.insert(0, current)
            current = previous_nodes[current]

        return path if path[0] == start_node_id else None


    def yen_k_shortest_paths(self, start_node_id, end_node_id, K=3):
        A = []  # Holds the k shortest paths
        B = []  # Candidate paths
        
        # Find the shortest path from start to end
        initial_path = self.dijkstra(start_node_id, end_node_id)
        if initial_path:
            A.append(initial_path)
        
        # Main loop for k = 1 to K
        for k in range(1, K):
            # For each node n in the k-1th shortest path
            for i in range(len(A[-1]) - 1):
                root_path = A[-1][:i+1]
                spur_node = A[-1][i]

                print("Spur Node:", spur_node)
                print("Root Path:", root_path)

                # Remove all nodes in root path from the graph
                self.remove_nodes(root_path[:-1])
                print("Removed Nodes:", self.removed_nodes)

                for path in A:
                    if path[:i+1] == root_path:
                        self.remove_edge(Edge(path[i], path[i+1], 0))

                # find spur paths
                spur_path = self.dijkstra(spur_node, end_node_id)
                print("Spur Path:", spur_path)

                # restore the nodes of the graph
                self.restore_nodes()
                self.restore_edges()
                print("Restored Nodes:", self.removed_nodes)

                if spur_path:
                    # Concatenate root and spur paths
                    total_path = root_path[:-1] + spur_path
                    # Add the new path to B
                    if total_path not in A and total_path not in B:
                        heappush(B, (total_path))

                print("Current A paths:", A)
                print("Current B paths:", B)


            # Add the shortest path in B to A
            if B:
                shortest_path_B = heappop(B)
                A.append(shortest_path_B)
                
        return A
    
    def path_cost(self, path):
        cost = 0
        for i in range(len(path) - 1):
            for edge in self.edges:
                if edge.start_node == path[i] and edge.end_node == path[i + 1]:
                    cost += edge.weight
                    break
        return cost

    def print_paths_with_costs(self, paths):
        for i, path in enumerate(paths):
            cost = self.path_cost(path)
            print(f"Path {i+1}: {path} with total cost {cost}")

    def get_node(self, node_id):
        """
        Retrieve information for a list of node IDs

        Parameters:
            - node_ids: node ID
        
        Return: 
            - node objects
        """
        node = self.nodes.get(node_id)

        return node
