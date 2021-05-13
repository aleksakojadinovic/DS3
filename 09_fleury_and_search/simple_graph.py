from collections import defaultdict
import copy 

class SimpleGraph:
    @staticmethod
    def dummy_callback(*args, **kwargs):
        return None

    @staticmethod
    def stack_pop(collection):
        x = collection.pop()
        return x, collection

    @staticmethod
    def queue_pop(collection):
        x = collection.pop(0)
        return x, collection

    EULER_HAS_PATH          = 0
    EULER_HAS_CYCLE         = 1
    EULER_HAS_NOTHING       = 2

    def __init__(self, num_nodes, **kwargs):
        self.num_nodes = num_nodes
        self.num_edges = 0
        self.adj_list = defaultdict(set)
        self.degrees  = [0 for _ in range(self.num_nodes)]

    def connect(self, node_from, node_to):
        if node_to in self.adj_list[node_from]:
            return
        self.adj_list[node_from].add(node_to)
        self.degrees[node_to] += 1
        self.adj_list[node_to].add(node_from)
        self.degrees[node_from] += 1

        self.num_edges += 1
    
    def disconnect(self, node_from, node_to, directed=False):
        if node_to not in self.adj_list[node_from]:
            return
        self.adj_list[node_from].remove(node_to)
        self.degrees[node_to] -= 1
        self.adj_list[node_to].remove(node_from)
        self.degrees[node_from] -= 1

        self.num_edges -= 1

    def get_active_nodes(self):
        return list(self.adj_list.keys())

    
    def euler_check(self):
        if not self.connected():
            return SimpleGraph.EULER_HAS_NOTHING


        num_odd_degrees = sum(map(lambda x: 0 if self.degrees[x] % 2 == 0 else 1, range(self.num_nodes)))
        if num_odd_degrees == 0:
            return SimpleGraph.EULER_HAS_CYCLE
        if num_odd_degrees == 2:
            return SimpleGraph.EULER_HAS_PATH

        return SimpleGraph.EULER_HAS_NOTHING

    def traversal_(self, start_node, callback, pop_func, ordered):
        
        if start_node < 0 or start_node >= self.num_nodes:
            raise ValueError(f'Invalid start node in traversal: {start_node}')
        
        nodes = [start_node]
        visited = [False for _ in range(self.num_nodes)]

        adj_list = copy.deepcopy(self.adj_list)
        if ordered:
            adj_list = [[] for i in range(self.num_nodes)]
            for key in sorted(list(self.adj_list.keys())):
                adj_list[key] = sorted(list(self.adj_list[key]))

        # print(adj_list)
        while nodes:

            node, nodes = pop_func(nodes)
            if visited[node]:
                continue
            visited[node] = True
            callback(node, visited)

            for x in adj_list[node]:
                if visited[x]:
                    continue
                nodes.append(x)

        return visited

    def connected(self):
        return all(self.dfs())


    def dfs(self, start_node=None, callback=None, ordered=False):
        if start_node is None:
            start_node = 0
        if callback is None:
            callback = SimpleGraph.dummy_callback
        return self.traversal_(start_node, callback, SimpleGraph.stack_pop, ordered)

    def bfs(self, start_node=None, callback=None, ordered=False):
        if start_node is None:
            start_node = 0
        if callback is None:
            callback = SimpleGraph.dummy_callback
        return self.traversal_(start_node, callback, SimpleGraph.queue_pop, ordered)




    def fleury(self):
        euler_start = self.get_euler_start_node()
        if euler_start is None:
            return None
