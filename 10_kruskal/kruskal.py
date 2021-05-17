import numpy as np

class Graph:

    def __init__(self, num_nodes) -> None:
        self.num_nodes = num_nodes
        self.adj_list = [[] for _ in range(num_nodes)]

    def connect(self, node_from, node_to, weight):
        if list(filter(lambda x: x[0] == node_to, self.adj_list[node_from])):
            return
        if list(filter(lambda x: x[0] == node_from, self.adj_list[node_to])):
            return
        self.adj_list[node_from].append((node_to, weight))
        self.adj_list[node_to].append((node_from, weight))

    def disconnect(self, node_from, node_to):
        self.adj_list[node_from] = list(filter(lambda x: x[0] != node_to, self.adj_list[node_from]))
        self.adj_list[node_to] = list(filter(lambda x: x[0] != node_from, self.adj_list[node_to]))

    # Returns a list of triples (int, int, float) meaning (node_from, node_to, weight)
    def get_edges(self):
        edges = set()
        for i in range(self.num_nodes):
            for j, w in self.adj_list[i]:
                if (i, j, w) not in edges and (j, i, w) not in edges:
                    edges.add((i, j, w))

        return list(edges)

  
def kruskal(graph):
    V = graph.num_nodes

    k = [i+1 for i in range(V)]
    L = []
    F = graph.get_edges()

    while F:
        (u, v, w) = min(F, key=lambda edge: edge[2])
        if k[u] != k[v]:
            max_idx = max(k[u], k[v])
            min_idx = min(k[u], k[v])
            for node_idx, component_idx in enumerate(k):
                if component_idx == max_idx:
                    k[node_idx] = min_idx
            
            L.append((u, v, w))
            
        F.remove((u, v, w))


    component_indices = set(k)
    for aps_idx, component_idx in enumerate(component_indices):
        nodes_in_component = [i for i, c in enumerate(k) if c == component_idx]
        edges_in_component = [(u, v, w) for u, v, w in L if u in nodes_in_component and v in nodes_in_component]
        print(f'Component {aps_idx + 1}')
        print(f'\t nodes: {nodes_in_component}')
        print(f'\t edges: {edges_in_component}')
    

    

def example1():
    g = Graph(7)
    g.connect(0, 1, 1)
    g.connect(1, 2, 3)
    g.connect(2, 3, 2)
    g.connect(3, 0, 3)
    g.connect(0, 2, 4)

    g.connect(4, 5, 4)
    g.connect(5, 6, 2)
    g.connect(4, 6, 3)

    kruskal(g)



if __name__ == '__main__':
    example1()
