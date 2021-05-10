from collections import defaultdict
import copy 

class SimpleGraph:
    @staticmethod
    def dummy_callback(*args, **kwargs):
        return None

    @staticmethod
    def stack_pop(collection):
        return collection.pop()

    @staticmethod
    def queue_pop(collection):
        return collection.pop(0)

    EULER_HAS_PATH          = 0
    EULER_HAS_CYCLE         = 1
    EULER_HAS_NOTHING       = 2

    def __init__(self, num_nodes, **kwargs):
        self.num_nodes = num_nodes
        self.num_edges = 0
        self.adj_list = defaultdict(set)
        self.degrees  = [0 for _ in range(self.num_nodes)]

    def connect(self, node_from, node_to, directed=False):
        self.adj_list[node_from].add(node_to)
        self.degrees[node_to] += 1
        self.num_edges += 1
        if not directed:
            self.adj_list[node_to].add(node_from)
            self.degrees[node_from] += 1
    
    def disconnect(self, node_from, node_to, directed=False):
        self.adj_list[node_from].remove(node_to)
        self.degrees[node_to] -= 1
        if not directed:
            self.adj_list[node_to].remove(node_from)
            self.degrees[node_from] -= 1

    def get_active_nodes(self):
        return list(self.adj_list.keys())

    
    def euler_check(self):
        num_odd_degrees = sum(map(lambda x: 0 if self.degrees[x] % 2 == 0 else 1, range(self.num_nodes)))
        if num_odd_degrees == 0:
            return SimpleGraph.EULER_HAS_CYCLE
        if num_odd_degrees == 2:
            return SimpleGraph.EULER_HAS_PATH

        return SimpleGraph.EULER_HAS_NOTHING

    def traversal_(self, start_node, callback, pop_func):
        
        if start_node < 0 or start_node >= self.num_nodes:
            raise ValueError(f'Invalid start node in traversal: {start_node}')
        
        nodes = [start_node]
        visited = [False for _ in range(self.num_nodes)]

        while nodes:
            
            node = pop_func(nodes)
            visited[node] = True
            callback(node, visited)

            for x in self.adj_list[node]:
                if visited[x]:
                    continue
                nodes.append(x)

        return visited

    def dfs(self, start_node=None, callback=None):
        if start_node is None:
            start_node = 0
        if callback is None:
            callback = SimpleGraph.dummy_callback
        return self.traversal_(start_node, callback, SimpleGraph.stack_pop)

    def bfs(self, start_node=None, callback=None):
        if start_node is None:
            start_node = 0
        if callback is None:
            callback = SimpleGraph.dummy_callback
        return self.traversal_(start_node, callback, SimpleGraph.queue_pop)


    def fleury(self):
        euler_start = self.get_euler_start_node()
        if euler_start is None:
            return None


def get_euler_start_node(graph):
    ec = graph.euler_check()
    if ec == SimpleGraph.EULER_HAS_NOTHING:
        return None

    if ec == SimpleGraph.EULER_HAS_CYCLE:
        # We can start anywhere
        return 0
    else:
        # We have to start at an odd node
        return [i for i, deg in enumerate(graph.degrees) if deg % 2 == 1][0]

# Picks an edge from node for Fleury's algorithm
def pick_edge(graph, node):
    candidate_edges = graph.adj_list[node]
    if len(candidate_edges) == 1:
        return list(candidate_edges)[0]

    for neighbor in candidate_edges:
        # Check whether this edge is a bridge
        tmp_graph = copy.deepcopy(graph)
        tmp_graph.disconnect(node, neighbor)
        visited = tmp_graph.dfs(start_node=node)
        if visited[neighbor]:
            return neighbor


    raise ValueError(f'Could not find appropriate edge!')

def fleury_algorithm(graph):
    start_node = get_euler_start_node(graph)
    if start_node is None:
        return None

    
    

    graph = copy.deepcopy(graph)

    V = graph.num_nodes
    E = graph.num_edges
    curr_path = [start_node]
    curr_node = start_node

    edges_hit = 0

    while True:

        if edges_hit == E:
            break
        
        node_to = pick_edge(graph, curr_node)
        graph.disconnect(curr_node, node_to)
        curr_path.append(node_to)
        curr_node = node_to
        edges_hit += 1

    return curr_path




def example1():
    g = SimpleGraph(4)
    g.connect(0, 1)
    g.connect(0, 2)
    g.connect(1, 2)
    g.connect(2, 3)
    print(fleury_algorithm(g))

def example2():
    g = SimpleGraph(5)
    g.connect(0, 1)
    g.connect(1, 2)
    g.connect(2, 0)
    print(fleury_algorithm(g))

def example3():
    g = SimpleGraph(5)
    g.connect(1, 0)
    g.connect(0, 2)
    g.connect(2, 1)
    g.connect(0, 3)
    g.connect(3, 4)
    g.connect(3, 2)
    g.connect(3, 1)
    g.connect(2, 4)
    print(fleury_algorithm(g))

if __name__ == '__main__':
    example3()
    
        
