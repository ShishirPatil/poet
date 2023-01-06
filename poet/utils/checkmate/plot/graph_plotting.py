import pathlib
from typing import Optional

import numpy as np
from graphviz import Digraph

from poet.utils.checkmate.core.dfgraph import DFGraph
from poet.utils.checkmate.core.schedule import ScheduledResult
from poet.utils.checkmate.core.utils.definitions import PathLike


def plot_dfgraph(g: DFGraph, directory, format="pdf", quiet=True, name=""):
    """Generate Graphviz-formatted edge list for visualization, and write pdf"""
    print("Plotting network architecture...")
    dot = Digraph("render_dfgraph" + str(name))
    dot.attr("graph")
    for u in g.v:
        node_name = g.node_names.get(u)
        node_name = node_name if node_name is None else "{} ({})".format(node_name, str(u))
        attrs = {"style": "filled"} if g.is_backward_node(u) else {}
        dot.node(str(u), node_name, **attrs)
    for edge in g.edge_list:
        dep_order = str(g.args[edge[-1]].index(edge[0]))
        dot.edge(*map(str, edge), label=dep_order)
    try:
        dot.render(directory=directory, format=format, quiet=quiet)
    except TypeError:
        dot.render(directory=directory, format=format)
    print("Saved network architecture plot to directory:", directory)


def plot_schedule(sched_result: ScheduledResult, plot_mem_usage=False, save_file: Optional[PathLike] = None, show=False, plt=None):
    assert sched_result.feasible
    R = sched_result.schedule_aux_data.R
    S = sched_result.schedule_aux_data.S
    U = None if sched_result.ilp_aux_data is None else sched_result.ilp_aux_data.U
    mem_grid = None if sched_result.schedule_aux_data is None else sched_result.schedule_aux_data.mem_grid
    _plot_schedule_from_rs(R, S, plot_mem_usage, mem_grid, U, save_file, show, plt)


def _plot_schedule_from_rs(R, S, plot_mem_usage=False, mem_grid=None, U=None, save_file: Optional[PathLike] = None, show=False, plt=None):
    if plt is None:
        import matplotlib.pyplot as plt

    if plot_mem_usage:
        assert mem_grid is not None
        fig, axs = plt.subplots(1, 4)
        vmax = mem_grid
        vmax = vmax if U is None else max(vmax, np.max(U))

        # Plot slow verifier memory usage
        axs[2].invert_yaxis()
        axs[2].pcolormesh(mem_grid, cmap="Greys", vmin=0, vmax=vmax)
        axs[2].set_title("Memory usage (verifier)")

        # Plot solver memory usage variables
        axs[3].invert_yaxis()
        axs[3].set_title("Memory usage (solved)")
        if U is not None:
            axs[3].pcolormesh(U, cmap="Greys", vmin=0, vmax=vmax)

        fig.set_size_inches(28, 6)
    else:
        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(18, 6)

    axs[0].invert_yaxis()
    axs[0].pcolormesh(R, cmap="Greys", vmin=0, vmax=1)
    axs[0].set_title("R")

    axs[1].invert_yaxis()
    axs[1].pcolormesh(S, cmap="Greys", vmin=0, vmax=1)
    axs[1].set_title("S")

    if show:
        plt.show()
    if save_file:
        path = pathlib.Path(save_file)
        path.parents[0].mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
