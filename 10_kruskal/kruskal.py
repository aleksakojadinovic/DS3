import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


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

    def to_networkx_graph(self):
        dod = dict()
        for i in range(self.num_nodes):
            dod[i] = dict()
            for j, w in self.adj_list[i]:
                dod[i][j] = {"weight": w}
        
        return nx.from_dict_of_dicts(dod)


def edge_str(u, v, w):
    return f'{u}--{w}--{v}'

def edge_str_t(t):
    u, v, w = t
    return edge_str(u, v, w)
  
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
    result = dict()
    result['components'] = dict()
    for aps_idx, component_idx in enumerate(component_indices):
        nodes_in_component = [i for i, c in enumerate(k) if c == component_idx]
        edges_in_component = [(u, v, w) for u, v, w in L if u in nodes_in_component and v in nodes_in_component]
        result['components'][aps_idx] = {"nodes": nodes_in_component, "edges": edges_in_component}
    result['all_edges'] = L
    return result

def run_kruskal(g, visualize=True):
    kruskal_res = kruskal(g)
    kruskal_all_edges = list(map(lambda x: (x[0], x[1]), kruskal_res['all_edges']))

    for k in kruskal_res["components"]:
        print(f'Component {k+1}')
        print('\tNodes:')
        print(f'\t\t{kruskal_res["components"][k]["nodes"]}')
        print(f'\tEdges:')
        edge_list = kruskal_res["components"][k]["edges"]
        for u, v, _ in edge_list:
            print(f'\t\t{u}----{v}')
        
    if not visualize:
        return

        
    nxg = g.to_networkx_graph()    
    nx_all_edges = nxg.edges()  

    edge_colors = ['r' if e in kruskal_all_edges else 'k' for e in nx_all_edges]
    edge_widths = [3 if e in kruskal_all_edges else 2 for e in nx_all_edges]

    
    pos = nx.circular_layout(nxg)
    labels = nx.get_edge_attributes(nxg,'weight')

    nx.draw(nxg, pos, with_labels=True, edge_color=edge_colors, width=edge_widths)
    nx.draw_networkx_edge_labels(nxg, pos, edge_labels=labels)

    plt.show()

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

    run_kruskal(g, visualize=False)

    


if __name__ == '__main__':
    example1()
