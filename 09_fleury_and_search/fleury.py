import numpy as np

from collections import defaultdict

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

    def __init__(self, num_nodes, **kwargs):
        self.num_nodes = num_nodes
        self.adj_list = defaultdict(set)
        self.degrees  = [0 for _ in range(self.num_nodes)]

    def connect(self, node_from, node_to, directed=False):
        self.adj_list[node_from].add(node_to)
        self.degrees[node_to] += 1
        if not directed:
            self.adj_list[node_to].add(node_from)
            self.degrees[node_from] += 1
    
    def disconnect(self, node_from, node_to, directed=False):
        self.adj_list[node_from].remove(node_to)
        self.degrees[node_to] -= 1
        if directed:
            self.adj_list[node_to].remove(node_from)
            self.degrees[node_from] -= 1

    def get_active_nodes(self):
        return list(self.adj_list.keys())

    def has_euler_cycle(self):
        num_even_degrees = sum(map(lambda x: 1 if self.degrees[x] % 2 == 0 else 0, self.degrees))
        return num_even_degrees == self.num_nodes

    def has_euler_cycle(self):
        num_even_degrees = sum(map(lambda x: 1 if self.degrees[x] % 2 == 0 else 0, self.degrees))
        num_odd_degrees = self.num_nodes - num_even_degrees
        return num_odd_degrees == 2

    def traversal_(self, start_node=None, callback=SimpleGraph.dummy_callback, pop_func=SimpleGraph.stack_pop):
        if start_node < 0 or start_node >= self.num_nodes:
            raise ValueError(f'Invalid start node in traversal: {start_node}')
        
        nodes = [start_node]
        visited = [False for _ in range(self.num_nodes)]

        while nodes:
            node = pop_func(nodes)
            callback(node)

            for x in self.adj_list[node]:
                if visited[x]:
                    continue
                nodes.append(x)

    def dfs(self, start_node=None, callback=SimpleGraph.dummy_callback):
        return self.traversal_(start_node, callback, SimpleGraph.stack_pop)

    def bfs(self, start_node, callback):
        return self.traversal_(start_node, callback, SimpleGraph.queue_pop)

        

        

        
        

                
            


    def bfs():
        pass





if __name__ == '__main__':
    pass
        
