import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from poet.utils.checkmate.core.dfgraph import DFGraph
from poet.utils.checkmate.core.graph_builder import GraphBuilder
from poet.utils.checkmate.core.utils.definitions import PathLike
from poet.power_computation import DNNLayer, get_net_costs, GradientLayer


def save_network_repr(net: List[DNNLayer], readable_path: PathLike = None, pickle_path: PathLike = None):
    if readable_path is not None:
        with Path(readable_path).open("w") as f:
            for layer in net:
                f.write("{}\n".format(layer))
    if pickle_path is not None:
        with Path(pickle_path).open("wb") as f:
            pickle.dump(net, f)


def make_dfgraph_costs(net, device):
    power_specs = get_net_costs(net, device)
    per_layer_specs = pd.DataFrame(power_specs).to_dict(orient="records")

    layer_names, power_cost_dict, page_in_cost_dict, page_out_cost_dict = {}, {}, {}, {}
    gb = GraphBuilder()
    for idx, (layer, specs) in enumerate(zip(net, per_layer_specs)):
        layer_name = "layer{}_{}".format(idx, layer.__class__.__name__)
        layer_names[layer] = layer_name
        gb.add_node(layer_name, cpu_cost=specs["runtime_ms"], ram_cost=specs["memory_bytes"], backward=isinstance(layer, GradientLayer))
        gb.set_parameter_cost(gb.parameter_cost + specs["param_memory_bytes"])
        page_in_cost_dict[layer_name] = specs["pagein_cost_joules"]
        page_out_cost_dict[layer_name] = specs["pageout_cost_joules"]
        power_cost_dict[layer_name] = specs["compute_cost_joules"]
        for dep in layer.depends_on:
            gb.add_deps(layer_name, layer_names[dep])
    g = gb.make_graph()

    ordered_names = [(topo_idx, name) for topo_idx, name in g.node_names.items()]
    ordered_names.sort(key=lambda x: x[0])
    ordered_names = [x for _, x in ordered_names]

    compute_costs = np.asarray([power_cost_dict[name] for name in ordered_names]).reshape((-1, 1))
    page_in_costs = np.asarray([page_in_cost_dict[name] for name in ordered_names]).reshape((-1, 1))
    page_out_costs = np.asarray([page_out_cost_dict[name] for name in ordered_names]).reshape((-1, 1))
    return g, compute_costs, page_in_costs, page_out_costs


def extract_costs_from_dfgraph(g: DFGraph, sd_card_multipler=5.0):
    T = g.size
    cpu_cost_vec = np.asarray([g.cost_cpu[i] for i in range(T)])[np.newaxis, :].T
    page_in_cost_vec = cpu_cost_vec * sd_card_multipler
    page_out_cost_vec = cpu_cost_vec * sd_card_multipler
    return cpu_cost_vec, page_in_cost_vec, page_out_cost_vec
