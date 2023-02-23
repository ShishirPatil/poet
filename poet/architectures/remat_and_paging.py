import torch.nn as nn


# output layer traversal path for given layer in model
def get_all_parent_layers(net, type):
    layers = []
    for name in net.named_modules():
        if name == type:
            layer = net
            attributes = name.strip().split(".")
            for attr in attributes:
                if not attr.isnumeric():
                    layer = getattr(layer, attr)
                else:
                    layer = layer[int(attr)]
            layers.append([layer, attributes[-1]])
    return layers


# implements rematerializaion and paging techniques in inputted model
# remat - sets inputted node to Identity layer and saves node and arguments
# paging - page out layer to cpu
def memory_saving(model_indexer, node, is_remat, is_page, remat_list):
    if is_remat:
        remat_list.append([node.target, getattr(model_indexer[0], model_indexer[1])])
        setattr(model_indexer[0], model_indexer[1], nn.Identity())
    elif is_page:
        layer = getattr(model_indexer[0], model_indexer[1])
        layer = layer.cpu()
    return


# remat - recomputes inputted node and sets layer back in model
# paging - page in layer to gpu
def reuse_layer(model_indexer, node, is_remat, is_page, remat_list):
    if is_remat:
        for layer in remat_list:
            if layer[0] == node.target:
                break
        setattr(model_indexer[0], model_indexer[1], layer[1])
    elif is_page:
        layer = getattr(model_indexer[0], model_indexer[1])
        layer = layer.gpu()
    return
