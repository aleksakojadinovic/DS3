"""
Implements the functionality of a directed graph.
"""
class DirectedGraph:

    def __init__(self, num_nodes) -> None:
        self.num_nodes          = num_nodes
        self.adj_list           = [[] for _ in range(num_nodes)]
        self.in_degrees         = [0 for _ in range(num_nodes)]
        self.out_degrees        = [0 for _ in range(num_nodes)]

    def connect(self, node_from: int, node_to: int, weight: float) -> None:
        if list(filter(lambda entry: entry[0] == node_to, self.adj_list[node_from], self.adj_list[node_from])):
            raise ValueError(f'Node {node_from} is already connected to node {node_to}.')

        self.adj_list[node_from].append((node_to, weight))
        self.in_degrees[node_to] += 1
        self.out_degrees[node_from] += 1

    def disconnect(self, node_from: int, node_to: int) -> None:
        new_list = list(filter(lambda entry: entry[0] != node_to, self.adj_list[node_from]))
        if len(new_list) == len(self.adj_list[node_from]):
            return
        self.adj_list[node_from] = new_list
        self.out_degrees[node_from] -= 1
        self.in_degrees[node_to] -= 1

    
        
    