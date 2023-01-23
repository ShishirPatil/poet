from poet.power_computation import LinearLayer, ReLULayer, CrossEntropyLoss, GradientLayer, InputLayer, get_net_costs


def make_linear_network(batch_size=1, input_shape=[784]):
    layers = [InputLayer((batch_size, *input_shape))]

    linear_layers = [[784, 10], [10, 120], [120, 100], [100, 200], [200, 10], [10, 10]]
    for in_dim, out_dim in linear_layers:
        lin_layer = LinearLayer(in_dim, out_dim, input=layers[-1])
        act_layer = ReLULayer(lin_layer)
        layers.extend([lin_layer, act_layer])
    layers.append(CrossEntropyLoss(layers[-1], n_classes=10))
    bwd_layers = list(reversed(layers))
    for out_layer, in_layer in zip(bwd_layers[1:], bwd_layers[:-1]):
        layers.append(GradientLayer(out_layer, [in_layer], layers[-1]))

    return layers


def make_unit_linear_network(nfwd=12):
    layers = [InputLayer((1,))]
    for i in range(nfwd):
        layers.append(ReLULayer(layers[-1]))
    bwd_layers = list(reversed(layers))
    for out_layer, in_layer in zip(bwd_layers[1:], bwd_layers[:-1]):
        layers.append(GradientLayer(out_layer, [in_layer], layers[-1]))
    return layers


if __name__ == "__main__":
    from poet.chipsets import *

    CHIPSET = MKR1000
    print("### network ###")
    net = make_linear_network()
    for layer in net:
        print(layer)

    # print("### Profiles ###")
    # resource = get_net_costs(net, CHIPSET)
    # for _list in resource:
    #     print(_list, resource[_list])
