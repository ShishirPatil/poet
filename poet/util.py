from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from poet.architectures.bert import BERTBase
from poet.architectures.linear import make_linear_network
from poet.architectures.resnet import resnet18, resnet18_cifar, resnet50
from poet.architectures.vgg import vgg16
from poet.chipsets import M4F, MKR1000, JetsonTX2, RPi, RPiNoCache
from poet.poet_solver import POETSolution
from poet.power_computation import DNNLayer, GradientLayer, get_net_costs
from poet.utils.checkmate.core.dfgraph import DFGraph
from poet.utils.checkmate.core.graph_builder import GraphBuilder
from poet.utils.checkmate.core.utils.definitions import PathLike
from poet.utils.checkmate.plot.graph_plotting import plot_dfgraph


@dataclass
class POETResult:
    ram_budget: float
    runtime_budget_ms: float
    paging: bool
    remat: bool
    total_power_cost_page: float
    total_power_cost_cpu: float
    total_runtime: float
    feasible: bool
    solution: POETSolution


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


def get_chipset_and_net(platform: str, model: str, batch_size: int, mem_power_scale: float = 1.0):
    if platform == "m0":
        chipset = MKR1000
    elif platform == "a72":
        chipset = RPi
    elif platform == "a72nocache":
        chipset = RPiNoCache
    elif platform == "m4":
        chipset = M4F
    elif platform == "jetsontx2":
        chipset = JetsonTX2
    else:
        raise NotImplementedError()

    chipset["MEMORY_POWER"] *= mem_power_scale

    if model == "linear":
        net = make_linear_network()
    elif model == "vgg16":
        net = vgg16(batch_size)
    elif model == "vgg16_cifar":
        net = vgg16(batch_size, 10, (3, 32, 32))
    elif model == "resnet18":
        net = resnet18(batch_size)
    elif model == "resnet50":
        net = resnet50(batch_size)
    elif model == "resnet18_cifar":
        net = resnet18_cifar(batch_size, 10, (3, 32, 32))
    elif model == "bert":
        net = BERTBase(SEQ_LEN=512, HIDDEN_DIM=768, I=64, HEADS=12, NUM_TRANSFORMER_BLOCKS=12)
    elif model == "transformer":
        net = BERTBase(SEQ_LEN=512, HIDDEN_DIM=768, I=64, HEADS=12, NUM_TRANSFORMER_BLOCKS=1)
    else:
        raise NotImplementedError()

    return chipset, net


def plot_network(
    platform: str, model: str, directory: str, batch_size: int = 1, mem_power_scale: float = 1.0, format="pdf", quiet=True, name=""
):
    chipset, net = get_chipset_and_net(platform, model, batch_size, mem_power_scale)
    g, *_ = make_dfgraph_costs(net, chipset)
    plot_dfgraph(g, directory, format, quiet, name)


def print_result(result: POETResult):
    solution = result.solution
    if solution.feasible:
        solution_msg = "successfully found an optimal solution" if solution.finished else "found a feasible solution"
        print(
            f"POET {solution_msg} with a memory budget of {result.ram_budget} bytes that consumes {result.total_power_cost_cpu:.5f} J of CPU power and {result.total_power_cost_page:.5f} J of memory paging power"
        )
        if not solution.finished:
            print("This solution is not guaranteed to be optimal - you can try increasing the time limit to find an optimal solution")

        plt.matshow(solution.R)
        plt.title("R")
        plt.show()

        plt.matshow(solution.SRam)
        plt.title("SRam")
        plt.show()

        plt.matshow(solution.SSd)
        plt.title("SSd")
        plt.show()
    else:
        print(
            "POET failed to find a feasible solution within the provided time limit. \n Either a) increase the memory and training time budgets, and/or b) increase the solve time [total_power_cost_page]"
        )
