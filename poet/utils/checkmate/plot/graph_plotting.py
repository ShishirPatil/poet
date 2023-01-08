import pathlib
from typing import Optional

import numpy as np
from graphviz import Digraph

from poet.utils.checkmate.core.dfgraph import DFGraph


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
