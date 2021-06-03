
import sys
import copy
from typing import List
import numpy as np
import pandas as pd
import argparse
import networkx as nx
import matplotlib.pyplot as plt



def fatal(msg):
    print(f'Fatal parse error: {msg}')
    sys.exit(1)

def read_lines_ds(filepath):
    try:
        lines = open(filepath, "r").readlines()
        lines = (line.strip() for line in lines)
        lines = (' '.join(line.split()) for line in lines)
        lines = (line for line in lines if line)
        lines = (line for line in lines if line[0] != '#')
        lines = list(lines)
        return lines
    except:
        fatal(f'Could not open graph file: {filepath}')
        

def parse_residual_network_new(filepath):
    lines = read_lines_ds(filepath)
    # V = None
    # try:
    #     V = int(lines[0])
    # except:
    #     fatal(f'Cannot parse number of nodes from {lines[0]}')

    network = ResidualNetworkGraph()
    for line in lines:
        line_strings = line.split(" ")
        if len(line_strings) != 3:
            fatal(f'Expecting three entries per line, got {line}')
        source      = line_strings[0]
        dest        = line_strings[1]
        cap_string  = line_strings[2]


        cap_val = None
        try:
            cap_val = int(cap_string)
        except:
            fatal(f'Failed to parse capacity from {cap_string}')

        network.add_edge(source, dest, 0, cap_val)
    

    return network
        


class ResidualNetworkGraph:
    def __init__(self) -> None:
        self.forward_edges  = []
        self.backward_edges = []
        self.nodes = set()


    def add_edge(self, node_from, node_to, flow, capacity):
        
        if flow >= capacity:
            raise ValueError(f'Flow cannot be greater than capacity.')

        existing_edge = self.fetch_by_source_and_dest(node_from, node_to, True)
        if existing_edge is not None:
            existing_edge["capacity"] += capacity
            existing_edge["flow"] += flow

        next_id = len(self.forward_edges)
        self.forward_edges.append({
            "node_from": node_from,
            "node_to": node_to,
            "flow": flow,
            "capacity": capacity,
            "forward": True,
            "id": next_id
        })

        self.backward_edges.append({
            "node_from": node_to,
            "node_to": node_from,
            "flow": 0,
            "capacity": 0,
            "forward": False,
            "id": next_id
        })

        self.nodes.add(node_from)
        self.nodes.add(node_to)

    
    def unpack_edge(self, edge):
        return edge["node_from"], edge["node_to"], edge["flow"], edge["capacity"], edge["forward"], edge["id"]

    def fetch_by_id(self, id, forward):
        return self.forward_edges[id] if forward else self.backward_edges[id]

    def fetch_by_source_and_dest(self, node_from, node_to, forward):
        if forward:
            col = [e for e in self.forward_edges if e["node_from"] == node_from and e["node_to"] == node_to]
        else:
            col = [e for e in self.backward if e["node_from"] == node_from and e["node_to"] == node_to]

        if not col:
            return None

        return col[0]

    # Converts it to a simple graph but it must keep track of which node is which
    def to_bfs_graph(self):
        node_list = list(sorted(self.nodes))
        graph = dict()

        # Initialize for every node
        for node in node_list:
            graph[node] = []

        for edge in self.forward_edges + self.backward_edges:
            node_from, node_to, flow, capacity, forward, id = self.unpack_edge(edge)

            # If an edge is fully saturated, we do not consider it at all
            if flow >= capacity:
                continue

            graph[node_from].append({
                "node_from": node_from,
                "node_to": node_to,
                "forward": forward,
                "id": id
            })

        for node in node_list:
            graph[node] = list(sorted(graph[node], key=lambda x: x["node_to"]))

        return graph, node_list




    def find_augmenting_path(self, source, sink):
        graph, node_list = self.to_bfs_graph()
    
        if source not in node_list:
            return None
        if sink not in node_list:
            return None

        
        q = [source]
        visited             = dict([(n, False) for n in node_list])
        parent              = dict([(n, None) for n in node_list])
        parent_edge_ids     = dict([(n, None) for n in node_list])

        while q:
            current_node = q.pop(0)
            if visited[current_node]:
                continue
            visited[current_node] = True
            if current_node == sink:
                current_node = sink
                edges_on_path = []
                while True:
                    if current_node == source:
                        return edges_on_path
                    
                    edges_on_path.append((
                        parent_edge_ids[current_node][0],
                        parent_edge_ids[current_node][1]
                    ))

                    current_node = parent[current_node]

                

            for next_entry in graph[current_node]:
                next_node, forward, edge_id = next_entry["node_to"], next_entry["forward"], next_entry["id"]
                if not visited[next_node] and next_node not in q:
                    parent[next_node] = current_node
                    parent_edge_ids[next_node] = (edge_id, forward)
                    q.append(next_node)

        return None


    def to_netx_graph(self):
        nx_graph = nx.DiGraph()

        for id, edge in enumerate(self.forward_edges):
            node_from, node_to, flow, capacity, _, _ = self.unpack_edge(edge)
            nx_graph.add_edge(node_from, node_to, edge_id=id, flow=flow, capacity=capacity)

        return nx_graph

def ek_net_graph(residual_network_graph: ResidualNetworkGraph, source, sink):
    nodes = residual_network_graph.nodes
    if source not in nodes:
        raise ValueError(f'Invalid source node: {source}')
    if sink not in nodes:
        raise ValueError(f'Invalid sink node: {sink}')

    while True:
        apath = residual_network_graph.find_augmenting_path(source, sink)

        if apath is None:
            break

        min_slack = None
        for edge_id, is_forward in apath:
            edge = residual_network_graph.fetch_by_id(edge_id, is_forward)
            _, _, flow, capacity, _, _ = residual_network_graph.unpack_edge(edge)

            slack = capacity - flow

            if min_slack is None or slack < min_slack:
                min_slack = slack


        for edge_id, is_forward in apath:
            edge = residual_network_graph.fetch_by_id(edge_id, is_forward)
            _, _, _, _, forward, _ = residual_network_graph.unpack_edge(edge)
            if forward:
                residual_network_graph.forward_edges[edge_id]["flow"] += min_slack
                residual_network_graph.backward_edges[edge_id]["flow"] -= min_slack
            else:
                residual_network_graph.backward_edges[edge_id]["flow"] -= min_slack
        

    flow = 0
    for edge in residual_network_graph.forward_edges:
        if edge["node_from"] != source:
            continue
        flow += edge["flow"]

    return flow




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--input',
                        required=True,
                        help='The input file.')

    parser.add_argument('-v',
                        '--visualize',
                        action='store_true',
                        help='Visualize the resulting flow.')

    parser.add_argument('-p',
                        '--prog',
                        default='neato',
                        help='Which graphviz visualization program to use.')

    parser.add_argument('-s',
                        '--source',
                        default='s',
                        help='The source node.')

    parser.add_argument('-e',
                        '--end',
                        default='t',
                        help='The sink node.')


    args = parser.parse_args()
    net = parse_residual_network_new(args.input)
    max_flow = ek_net_graph(net, args.source, args.end)

    print(f'Max flow found: ')
    print(f'\t{max_flow}')

    if args.visualize:
        nx_graph = net.to_netx_graph()
        pos = nx.nx_pydot.graphviz_layout(nx_graph, prog=args.prog)

        nx_graph_edges = nx_graph.edges(data=True)
        
        edge_labels = dict([      (   (u,v,),     f'{d["flow"]} / {d["capacity"]}')          for u,v,d in nx_graph_edges])
        edge_colors = ['blue' if d['flow'] > 0 else 'black' for _, _, d in nx_graph_edges]
        edge_widths = [2 + (d["flow"]/d["capacity"]) * 3 if d["capacity"] > 0 else 1 for _, _, d in nx_graph_edges]
        node_colors = ['darkorange' if x == args.source or x == args.end  else 'aqua' for x in nx_graph.nodes()]
    

        nx.draw(nx_graph, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors, width=edge_widths, connectionstyle='arc3, rad=0.1' )
        nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels)
        plt.show()



    
    

    




