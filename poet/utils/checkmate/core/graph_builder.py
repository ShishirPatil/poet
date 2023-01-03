import uuid
from typing import Dict, List, Set

from toposort import toposort

from poet.utils.checkmate.core.dfgraph import DFGraph
from poet.utils.checkmate.core.utils.graph import edge_to_adj_list


class GraphBuilder:
    def __init__(self):
        # map of user-facing names to internal uuid
        self.nodes = {}  # type: Dict[str, uuid.UUID]
        # set of backwards node
        self.backward_nodes = set()  # type: Set[uuid.UUID]
        # map of internal node uuid to its list of arguments
        self.arguments = {}  # type: Dict[uuid.UUID, List[uuid.UUID]]
        # map of per-node CPU costs
        self.costs_cpu = {}  # type: Dict[uuid.UUID, int]
        # map of per-node RAM costs
        self.costs_ram = {}  # type: Dict[uuid.UUID, int]
        # store total cost of parameters which is passed along to DFGraph initialization
        self.parameter_cost = 0

    def _name_to_uuid(self, name: str) -> uuid.UUID:
        if name not in self.nodes.keys():
            self.nodes[name] = uuid.uuid4()
        return self.nodes[name]

    def _uuid_to_name(self, uuid: uuid.UUID) -> str:
        """convenience method that is quite slow"""
        for name, idx in self.nodes.items():
            if idx == uuid:
                return name

    def set_parameter_cost(self, cost: int) -> "GraphBuilder":
        """
        Set graph-wide parameter costs which is used internally to adjust the budget to avoid OOM due to params
        :param cost: Cost in bytes to store parameters for this graph
        """
        self.parameter_cost = cost
        return self

    def add_node(self, name: str, cpu_cost: int, ram_cost: int, backward: bool = False) -> "GraphBuilder":
        """
        Add a node to the graph with associated graph properties
        :param name: A name for the node that is added to the graph
        :param cpu_cost: Cost to evaluate this node in runtime or in FLOPs
        :param ram_cost: RAM cost needed to store the output of this node along with the temporary state like workspace
        :param backward: Boolean denoting whether this node is a backward node
        """
        # TODO (paras) support flag denoting do-not-recompute
        assert name not in self.nodes.keys(), "Duplicate node name!"
        uuid = self._name_to_uuid(name)
        self.costs_cpu[uuid] = cpu_cost
        self.costs_ram[uuid] = ram_cost
        self.arguments[uuid] = []
        if backward:
            self.backward_nodes.add(uuid)
        return self

    def add_deps(self, dest_node: str, *source_nodes: str) -> "GraphBuilder":
        """
        Add dependencies to the adjacency list. This command will append listed sources in order that they are passed
        and add said dependencies after any existing nodes.
        :param dest_node: name for the destination node
        :param source_nodes: one or more names for any dependencies for the destination node
        """
        dest_node_uuid = self._name_to_uuid(dest_node)
        prior_nodes = self.arguments.get(dest_node_uuid, [])
        self.arguments[dest_node_uuid] = prior_nodes + list(map(self._name_to_uuid, source_nodes))
        return self

    def make_graph(self) -> DFGraph:
        """
        Build the DFGraph given the dependency structure described in the problem
        :return: a DFGraph instance corresponding to your problem description
        """
        # step 1 -- toposort graph and allocate node positions as a dict({0, ..., n} -> UUID)
        edge_list = [(source, dest) for dest, sources in self.arguments.items() for source in sources]
        topo_order = list(reversed([x for st in toposort(edge_to_adj_list(edge_list)) for x in st]))
        topo_order = [v for v in set(self.nodes.values()) - set(topo_order)] + topo_order  # add isolated nodes
        uuid2topo = {uuid: topo_idx for topo_idx, uuid in enumerate(topo_order)}

        # step 2 -- map builder data-structures to node position indexed data-structures
        vertex_list = list(uuid2topo.values())
        cost_cpu = dict((uuid2topo[idx], self.costs_cpu[idx]) for idx in self.costs_cpu.keys())
        cost_ram = dict((uuid2topo[idx], self.costs_ram[idx]) for idx in self.costs_ram.keys())
        arg_list = {uuid2topo[key]: [uuid2topo[arg] for arg in args] for key, args in self.arguments.items()}
        names = {uuid2topo[idx]: name for (name, idx) in self.nodes.items()}
        bwd_node_set = set(uuid2topo[v] for v in self.nodes.values() if v in self.backward_nodes)

        # step 3 -- make DFGraph
        return DFGraph(
            v=vertex_list,
            args=arg_list,
            backward_nodes=bwd_node_set,
            node_names=names,
            cost_cpu=cost_cpu,
            cost_ram=cost_ram,
            cost_ram_parameters=self.parameter_cost,
        )


def gen_linear_graph(forward_node_count) -> DFGraph:
    """
    gen_linear_graph will generate linear-style graphs like VGG and AlexNet.
    Method returns forward and backward graphs. Pass cost_ram and cost_cpu as kwargs.
    :param forward_node_count: number of forward (not backward nodes)
    :return: Graph object containing linear graph
    """
    gb = GraphBuilder()
    for i in range(forward_node_count * 2 + 1):
        gb.add_node("node{}".format(i), cpu_cost=1, ram_cost=1, backward=(i >= forward_node_count))
        if i > 0:
            gb.add_deps("node{}".format(i), "node{}".format(i - 1))

    for i in range(forward_node_count):
        corresponding_bwd = (forward_node_count * 2) - i
        gb.add_deps("node{}".format(corresponding_bwd), "node{}".format(i))
    return gb.make_graph()
