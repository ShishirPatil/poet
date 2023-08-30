from poet.power_computation import (
    Conv2dLayer,
    ReLULayer,
    MaxPool2d,
    LinearLayer,
    DropoutLayer,
    CrossEntropyLoss,
    GradientLayer,
    InputLayer,
    FlattenLayer,
    get_net_costs,
)


def vgg16(batch_size, num_classes=1000, input_shape=(3, 224, 224)):
    """
    Constructs a VGG-16 model.

    Parameters:
        batch_size (int): Size of the input batch.
        num_classes (int, optional): Number of output classes. Default is 1000.
        input_shape (tuple, optional): Shape of the input data. Default is (3, 224, 224).

    Returns:
        list: List of model layers.
    """
    def make_conv_stack(in_channels, out_filters, kernel_size, x):
        """
        Constructs a convolutional stack for the VGG-16 model.

        Parameters:
            in_channels (int): Number of input channels.
            out_filters (int): Number of output filters.
            kernel_size (tuple): Size of the convolutional kernel.
            x (Tensor): Input tensor.

        Returns:
            list: List of convolutional stack layers.
        """
        # Stride=1, padding=(1,1)
        conv = Conv2dLayer(in_channels, out_filters, kernel_size, 1, (1, 1), x)
        relu = ReLULayer(conv)
        return [conv, relu]

    layers = [InputLayer((batch_size, *input_shape))]
    layers.extend(make_conv_stack(3, 64, (3, 3), layers[-1]))
    layers.extend(make_conv_stack(64, 64, (3, 3), layers[-1]))
    layers.append(MaxPool2d((2, 2), 2, layers[-1]))  # 112
    layers.extend(make_conv_stack(64, 128, (3, 3), layers[-1]))
    layers.extend(make_conv_stack(128, 128, (3, 3), layers[-1]))
    layers.append(MaxPool2d((2, 2), 2, layers[-1]))  # 56
    layers.extend(make_conv_stack(128, 256, (3, 3), layers[-1]))
    layers.extend(make_conv_stack(256, 256, (3, 3), layers[-1]))
    layers.extend(make_conv_stack(256, 256, (3, 3), layers[-1]))
    layers.append(MaxPool2d((2, 2), 2, layers[-1]))  # 28
    layers.extend(make_conv_stack(256, 512, (3, 3), layers[-1]))
    layers.extend(make_conv_stack(512, 512, (3, 3), layers[-1]))
    layers.extend(make_conv_stack(512, 512, (3, 3), layers[-1]))
    layers.append(MaxPool2d((2, 2), 2, layers[-1]))  # 14
    layers.extend(make_conv_stack(512, 512, (3, 3), layers[-1]))
    layers.extend(make_conv_stack(512, 512, (3, 3), layers[-1]))
    layers.extend(make_conv_stack(512, 512, (3, 3), layers[-1]))
    layers.append(MaxPool2d((2, 2), 2, layers[-1]))  # 7 => [W X H] = [7 X 7]
    layers.append(FlattenLayer(layers[-1]))
    layers.append(LinearLayer(512 * input_shape[-2] / 32 * input_shape[-1] / 32, 4096, layers[-1]))
    layers.append(ReLULayer(layers[-1]))
    layers.append(DropoutLayer(layers[-1]))
    layers.append(LinearLayer(4096, 4096, layers[-1]))
    layers.append(ReLULayer(layers[-1]))
    layers.append(DropoutLayer(layers[-1]))
    layers.append(LinearLayer(4096, num_classes, layers[-1]))
    layers.append(CrossEntropyLoss(layers[-1], num_classes))

    bwd_layers = list(reversed(layers))
    for out_layer, in_layer in zip(bwd_layers[1:], bwd_layers[:-1]):
        layers.append(GradientLayer(out_layer, [in_layer], layers[-1]))

    return layers


if __name__ == "__main__":
    from poet.chipsets import *

    CHIPSET = RPi
    print("### network ###")
    net = vgg16(1, 10, (3, 32, 32))
    param_count = 0
    totalEnergy = 0
    totalRAM = 0
    for layer in net:
        param_count += layer.param_count
        totalEnergy += layer.energy(CHIPSET)
        totalRAM += layer.param_ram_usage(CHIPSET)
        print(layer)

    print("### Total number of parameters in network:", param_count)
    print("### Total energy (forward + backward) cost is:", totalEnergy)

    # print("### Profiles ###")
    # resource = get_net_costs(net, MKR1000)
    # for _list in resource:
    #     print(_list, resource[_list])
