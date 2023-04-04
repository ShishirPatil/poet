import torch.nn as nn


# output layer traversal path in model for passed-in layer
def get_all_parent_layers(net, type):
    layers = []
    # iterates over all layers in model
    for name in net.named_modules():
        # check if curent module name matches specified type
        if name == type:
            layer = net
            # extracts layer type from its name in the model which contains indices into sub-modules
            attributes = name.strip().split(".")
            for attr in attributes:
                if not attr.isnumeric():
                    # retrieve layer attribute
                    layer = getattr(layer, attr)
                else:
                    # index into sub-modules list traversing model's path of layers
                    layer = layer[int(attr)]
            # append list of final layer and attribute name
            layers.append([layer, attributes[-1]])
    return layers


# implements rematerializaion technique on inputted model
# which saves passed-in node during forward pass for later recomputation
# during the backward pass of model
def memory_saving(model_indexer, node, remat_list):
    # saves node and arguments for later recomputation
    remat_list.append([node.target, getattr(model_indexer[0], model_indexer[1])])
    # sets inputted node to Identity layer
    setattr(model_indexer[0], model_indexer[1], nn.Identity())
    return


# recomputes inputted node which was rematerialized and sets layer back into model
def reuse_layer(model_indexer, node, remat_list):
    # iterates over rematerialized nodes to find matching layer
    for layer in remat_list:
        if layer[0] == node.target:
            break
    # sets inputted node back to its original state
    setattr(model_indexer[0], model_indexer[1], layer[1])
    return
