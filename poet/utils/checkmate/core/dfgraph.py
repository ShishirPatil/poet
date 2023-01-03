from collections import defaultdict
from functools import lru_cache
from typing import Iterable, Dict, List, Set

from toposort import toposort
from poet.utils.checkmate.core.utils.definitions import Vertex, EdgeList, AdjList
from poet.utils.checkmate.core.utils.graph import edge_to_adj_list, adj_to_edge_list, gcd, connected_components


class DFGraph:
    def __init__(
        self,
        args: AdjList,
        v: Iterable[Vertex],
        backward_nodes: Iterable[Vertex] = None,
        cost_cpu: Dict[Vertex, int] = None,
        cost_ram: Dict[Vertex, int] = None,
        node_names: Dict[Vertex, str] = None,
        cost_ram_parameters: int = 0,
    ):
        """
        Graph defines the forward and backward graph for a neural network
        :param args (Dict[int, List[int]]): Dependency listing, where arguments ordered
        :param v: List of nodes in the graph
        :param cost_cpu: Dictionary mapping nodes to respective integral runtime costs (for forward/backward operators)
        :param cost_ram: Dictionary mapping nodes to respective integral memory costs (for forward/backward operators)
        """
        self.v = list(sorted(v))
        self.backward_nodes = set(backward_nodes) if backward_nodes else set()
        self.args = defaultdict(list, args)
        self.node_names = node_names if node_names is not None else {}
        self.cost_cpu = cost_cpu if cost_cpu else {v: 1 for v in set(self.v)}
        self.cost_ram = cost_ram if cost_ram else {v: 1 for v in set(self.v)}
        self.cost_ram_parameters = cost_ram_parameters

    @property
    @lru_cache(maxsize=1)
    def vfwd(self):
        return list(filter(lambda x: x not in self.backward_nodes, self.v))

    @property
    @lru_cache(maxsize=1)
    def vbwd(self):
        return list(filter(lambda x: x in self.backward_nodes, self.v))

    @property
    @lru_cache(maxsize=1)
    def adj_list(self):
        adj_list = defaultdict(list)
        for v, sources in self.args.items():
            for u in sources:
                adj_list[u].append(v)
        return dict(adj_list)

    @property
    @lru_cache(maxsize=1)
    def adj_list_fwd(self):
        adj_list = defaultdict(list)
        for v, sources in self.args.items():
            for u in sources:
                if v in self.vfwd:
                    adj_list[u].append(v)
        return dict(adj_list)

    @property
    @lru_cache(maxsize=1)
    def edge_list(self):
        return adj_to_edge_list(self.adj_list)

    @property
    @lru_cache(maxsize=1)
    def edge_list_fwd(self):
        return adj_to_edge_list(self.adj_list_fwd)

    @property
    def size(self):
        return len(self.v)

    @property
    def cost_ram_fixed(self):
        """Get fixed memory costs for the model (parameters and their gradients)"""
        return int(2 * self.cost_ram_parameters)

    def ram_gcd(self, *othervals):
        values = set(self.cost_ram.values()) | set(othervals)  # + [self.cost_ram_fixed]
        return gcd(*values)

    def cpu_gcd(self, *othervals):
        values = set(self.cost_cpu.values()) | set(othervals)
        return gcd(*values)

    @property
    @lru_cache(maxsize=1)
    def articulation_points(self) -> Set[Vertex]:
        """Determine checkpointable nodes in a forward graph (not backward)"""
        E = list(self.edge_list_fwd)
        V = set([i for (i, j) in E] + [j for (i, j) in E]).union({-1, -2})  # directed to undirected graph
        E = [(i, j) for (i, j) in E if i in V and j in V] + [(-1, 0), (max(V), -2)]
        checkpoint_ok = set()
        for v in filter(lambda v: v >= 0, V):  # ignore placeholders for input and output
            # count connected components in induced subgraph F = G / v
            Eprime = {e for e in E if v not in e}
            n_components = len(list(connected_components(edge_to_adj_list(Eprime, convert_undirected=True))))
            if n_components > 1:
                checkpoint_ok.add(v)
        return checkpoint_ok

    @property
    @lru_cache(maxsize=1)
    def topological_order(self):
        adj_set = {k: set(v) for k, v in self.adj_list.items()}
        topo_sets = list(toposort(adj_set))
        return [x for topo_set in topo_sets for x in topo_set]

    @property
    @lru_cache(maxsize=1)
    def topological_order_fwd(self):
        adj_set = {k: set(v) for k, v in self.adj_list_fwd.items()}
        topo_sets = list(toposort(adj_set))
        return [x for topo_set in topo_sets for x in topo_set]

    @property
    @lru_cache(maxsize=1)
    def _predecessor_dict(self):
        preds = defaultdict(list)
        for eidx, (u, v) in enumerate(self.edge_list):
            preds[v].append((eidx, u))
        return preds

    @property
    @lru_cache(maxsize=None)
    def _successor_dict(self):
        sucs = defaultdict(list)
        for eidx, (u, v) in enumerate(self.edge_list):
            sucs[u].append((eidx, v))
        return sucs

    def predecessors(self, node) -> Set[Vertex]:
        return {u for (_, u) in self._predecessor_dict[node]}

    def successors(self, node) -> Set[Vertex]:
        return {u for (_, u) in self._successor_dict[node]}

    def predecessors_indexed(self, node):
        return self._predecessor_dict[node]

    def successors_indexed(self, node):
        return self._successor_dict[node]

    def induce_subgraph(self, nodes: List[Vertex]) -> EdgeList:
        return [e for e in self.edge_list if all([x in nodes for x in e])]

    def is_forward_node(self, node: Vertex):
        return node not in self.backward_nodes

    def is_backward_node(self, node: Vertex):
        return node in self.backward_nodes

    @property
    def max_degree_ram(self):
        """compute minimum memory needed for any single node (ie inputs and outputs)"""
        vfwd = [v for v in self.v if v not in self.backward_nodes]
        return max([sum([self.cost_ram[u] for u in self.predecessors(v)]) + self.cost_ram[v] for v in vfwd])
