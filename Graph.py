class Graph:
    def __init__(self):
        self.nodes = {}  # Node ID -> Node object
        self.edges = []  # List of Edge objects

    def add_node(self, node):
        self.nodes[node.node_id] = node

    def add_edge(self, edge):
        self.edges.append(edge)