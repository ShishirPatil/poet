from poet.power_computation import (
    LinearLayer,
    ReLULayer,
    Conv2dLayer,
    FlattenLayer,
    TanHLayer,
    SigmoidLayer,
    SkipAddLayer,
    DropoutLayer,
    GradientLayer,
    InputLayer,
    SkipAddLayer,
    CrossEntropyLoss,
    GradientLayer,
    BatchNorm2d,
    MaxPool2d,
    AvgPool2d,
    GlobalAvgPool,
    get_net_costs,
)


# transforms input model's graph to output graph with POET layer nodes
def graph_transform(traced):
    for n in traced.graph.nodes:
        if "<built-in function" in str(n.target):
            continue
        elif "fc" in str(n.target):
            with traced.graph.inserting_after(n):
                new_node = traced.graph.call_function(LinearLayer, n.args, n.kwargs)
                n.replace_all_uses_with(new_node)
            traced.graph.erase_node(n)
        elif "flatten" in str(n.target):
            with traced.graph.inserting_after(n):
                new_node = traced.graph.call_function(FlattenLayer, n.args, n.kwargs)
                n.replace_all_uses_with(new_node)
            traced.graph.erase_node(n)
        elif "relu" in str(n.target):
            with traced.graph.inserting_after(n):
                new_node = traced.graph.call_function(ReLULayer, n.args, n.kwargs)
                n.replace_all_uses_with(new_node)
            traced.graph.erase_node(n)
        elif "conv" in str(n.target):
            with traced.graph.inserting_after(n):
                new_node = traced.graph.call_function(Conv2dLayer, n.args, n.kwargs)
                n.replace_all_uses_with(new_node)
            traced.graph.erase_node(n)
        elif "bn" in str(n.target):
            with traced.graph.inserting_after(n):
                new_node = traced.graph.call_function(BatchNorm2d, n.args, n.kwargs)
                n.replace_all_uses_with(new_node)
            traced.graph.erase_node(n)
        elif "maxpool" in str(n.target):
            with traced.graph.inserting_after(n):
                new_node = traced.graph.call_function(MaxPool2d, n.args, n.kwargs)
                n.replace_all_uses_with(new_node)
            traced.graph.erase_node(n)
    traced.recompile()
    return traced
