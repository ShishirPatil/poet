import fractions
from collections import defaultdict
from functools import reduce

from poet.utils.checkmate.core.utils.definitions import EdgeList, AdjList

import numpy as np


def edge_to_adj_list(E: EdgeList, convert_undirected=False):
    """Returns an (undirected / bidirectional) adjacency list"""
    adj_list = defaultdict(set)
    for (i, j) in E:
        adj_list[i].add(j)
        if convert_undirected:
            adj_list[j].add(i)
    return dict(adj_list)


def adj_to_edge_list(E: AdjList, convert_undirected=False, reverse_edge=False):
    """Returns an edge list
    :param E: AdjList -- input graph
    :param convert_undirected: bool -- if true, add u -> v and v -> u to output graph
    :param reverse_edge: bool -- if true, reverse edge direction prior to conversion
    :return:
    """
    edge_list = []
    for u, deps in E.items():
        for v in deps:
            edge = (u, v) if not reverse_edge else (v, u)
            edge_list.append(edge)
            if convert_undirected:
                edge_list.append(tuple(reversed(edge)))
    return edge_list


def connected_components(adj_list: AdjList):
    seen = set()

    def component(node):
        nodes = {node}
        while nodes:
            node = nodes.pop()
            seen.add(node)
            nodes |= set(adj_list[node]) - seen
            yield node

    for node in adj_list:
        if node not in seen:
            yield component(node)


def gcd(*args):
    values = np.array(list(args))
    intvalues = values.astype(np.int)
    if not np.allclose(intvalues, values):  # GCD is 1 if values are not integral
        return 1
    return reduce(fractions.gcd, intvalues)
