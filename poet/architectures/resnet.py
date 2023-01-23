from poet.power_computation import (
    AvgPool2d,
    BatchNorm2d,
    Conv2dLayer,
    CrossEntropyLoss,
    FlattenLayer,
    GradientLayer,
    InputLayer,
    LinearLayer,
    MaxPool2d,
    ReLULayer,
    SkipAddLayer,
    get_net_costs,
)

# Resnet implemented from the paper, not from PyTorch.


def resnet18(batch_size, num_classes=1000, input_shape=(3, 224, 224)):  # Imagenet
    def make_basic_block(in_planes, planes, stride, padding, x):
        kernel_size = (3, 3)
        conv1 = Conv2dLayer(in_planes, planes, kernel_size, stride, padding, x)
        bn1 = BatchNorm2d(conv1)
        relu1 = ReLULayer(bn1)
        conv2 = Conv2dLayer(planes, planes, kernel_size, 1, padding, relu1)
        bn2 = BatchNorm2d(conv2)
        skip = SkipAddLayer(relu1, bn2)
        relu2 = ReLULayer(skip)
        return [conv1, bn1, relu1, conv2, bn2, skip, relu2]

    layers = [InputLayer((batch_size, *input_shape))]
    # Input layers
    layers.append(Conv2dLayer(3, 64, (7, 7), 2, (3, 3), layers[-1]))  # => 112 X 112
    layers.append(BatchNorm2d(layers[-1]))
    # Block 1
    layers.append(MaxPool2d((3, 3), 2, layers[-1]))  # => 56 X 56
    layers.extend(make_basic_block(64, 64, 1, (1, 1), layers[-1]))
    layers.extend(make_basic_block(64, 64, 1, (1, 1), layers[-1]))
    # Block 2
    layers.extend(make_basic_block(64, 128, 2, (1, 1), layers[-1]))  # 28 X 28
    layers.extend(make_basic_block(128, 128, 1, (1, 1), layers[-1]))
    # Block 3
    layers.extend(make_basic_block(128, 256, 2, (1, 1), layers[-1]))  # 14 X 14
    layers.extend(make_basic_block(256, 256, 1, (1, 1), layers[-1]))
    # Block 4
    layers.extend(make_basic_block(256, 512, 2, (1, 1), layers[-1]))  # 7 X 7
    layers.extend(make_basic_block(512, 512, 1, (1, 1), layers[-1]))
    # Average pool
    layers.append(AvgPool2d((7, 7), 1, layers[-1]))  # 1 X 1 X 512
    layers.append(FlattenLayer(layers[-1]))

    layers.append(LinearLayer(512, num_classes, layers[-1]))
    # End of network. Find loss and reverse network.
    layers.append(CrossEntropyLoss(layers[-1], num_classes))
    bwd_layers = list(reversed(layers))
    for out_layer, in_layer in zip(bwd_layers[1:], bwd_layers[:-1]):
        layers.append(GradientLayer(out_layer, [in_layer], layers[-1]))

    return layers


def resnet50(batch_size, num_classes=1000, input_shape=(3, 224, 224)):  # Imagenet
    def make_basic_block(in_planes, planes_1, planes_2, stride, padding, x):
        # stride is 2 only for the first block, because downsampling
        conv1 = Conv2dLayer(in_planes, planes_1, (1, 1), stride, (0, 0), x)
        bn1 = BatchNorm2d(conv1)
        relu1 = ReLULayer(bn1)
        # Only the middle layer of each block is padded. Rest are not padded.
        conv2 = Conv2dLayer(planes_1, planes_1, (3, 3), 1, padding, relu1)
        bn2 = BatchNorm2d(conv2)
        relu2 = ReLULayer(bn2)
        conv3 = Conv2dLayer(planes_1, planes_2, (1, 1), 1, (0, 0), relu2)
        bn3 = BatchNorm2d(conv3)
        # Note that here relu1 and bn3 are not same size.
        # extra zeros get appended to relu1 to equal size of bn3. [Sec3.3. of paper]
        skip = SkipAddLayer(bn3, relu1)
        relu3 = ReLULayer(skip)
        return [conv1, bn1, relu1, conv2, bn2, relu2, conv3, bn3, skip, relu3]

    layers = [InputLayer((batch_size, *input_shape))]
    # Input layers
    layers.append(Conv2dLayer(3, 64, (7, 7), 2, (3, 3), layers[-1]))  # => 112 X 112
    layers.append(BatchNorm2d(layers[-1]))
    # Block 1
    layers.append(MaxPool2d((3, 3), 2, layers[-1]))  # => 56 X 56
    layers.extend(make_basic_block(64, 64, 256, 1, (1, 1), layers[-1]))
    layers.extend(make_basic_block(256, 64, 256, 1, (1, 1), layers[-1]))
    layers.extend(make_basic_block(256, 64, 256, 1, (1, 1), layers[-1]))
    # Block 2
    layers.extend(make_basic_block(256, 128, 512, 2, (1, 1), layers[-1]))  # 28 X 28
    layers.extend(make_basic_block(512, 128, 512, 1, (1, 1), layers[-1]))
    layers.extend(make_basic_block(512, 128, 512, 1, (1, 1), layers[-1]))
    layers.extend(make_basic_block(512, 128, 512, 1, (1, 1), layers[-1]))
    # Block 3
    layers.extend(make_basic_block(512, 256, 1024, 2, (1, 1), layers[-1]))  # 14 X 14
    layers.extend(make_basic_block(1024, 256, 1024, 1, (1, 1), layers[-1]))
    layers.extend(make_basic_block(1024, 256, 1024, 1, (1, 1), layers[-1]))
    layers.extend(make_basic_block(1024, 256, 1024, 1, (1, 1), layers[-1]))
    layers.extend(make_basic_block(1024, 256, 1024, 1, (1, 1), layers[-1]))
    layers.extend(make_basic_block(1024, 256, 1024, 1, (1, 1), layers[-1]))
    # Block 4
    layers.extend(make_basic_block(1024, 512, 2048, 2, (1, 1), layers[-1]))  # 7 X 7
    layers.extend(make_basic_block(2048, 512, 2048, 1, (1, 1), layers[-1]))
    layers.extend(make_basic_block(2048, 512, 2048, 1, (1, 1), layers[-1]))
    # Average pool
    layers.append(AvgPool2d((7, 7), 1, layers[-1]))  # 1 X 1 X 512
    layers.append(FlattenLayer(layers[-1]))
    layers.append(LinearLayer(2048, num_classes, layers[-1]))
    # End of network. Find loss and reverse network.
    layers.append(CrossEntropyLoss(layers[-1], num_classes))
    bwd_layers = list(reversed(layers))
    for out_layer, in_layer in zip(bwd_layers[1:], bwd_layers[:-1]):
        layers.append(GradientLayer(out_layer, [in_layer], layers[-1]))

    return layers


def resnet18_cifar(batch_size, num_classes=10, input_shape=(3, 32, 32)):  # CIFAR-10
    def make_basic_block(in_planes, planes, stride, padding, x):
        kernel_size = (3, 3)
        conv1 = Conv2dLayer(in_planes, planes, kernel_size, stride, padding, x)
        bn1 = BatchNorm2d(conv1)
        relu1 = ReLULayer(bn1)
        conv2 = Conv2dLayer(planes, planes, kernel_size, 1, padding, relu1)
        bn2 = BatchNorm2d(conv2)
        relu2 = ReLULayer(bn2)

        conv3 = Conv2dLayer(in_planes, planes, kernel_size, stride, padding, x)
        bn3 = BatchNorm2d(conv3)
        skip = SkipAddLayer(relu2, bn3)
        relu3 = ReLULayer(skip)
        return [conv1, bn1, relu1, conv2, bn2, relu2, conv3, bn3, skip, relu3]

    layers = [InputLayer((batch_size, *input_shape))]
    # Input layers
    layers.append(Conv2dLayer(3, 16, (3, 3), 1, (1, 1), layers[-1]))  # => 3 (channel) X 32 (H) X 32 (W)
    layers.append(BatchNorm2d(layers[-1]))
    # Block 1
    layers.extend(make_basic_block(16, 16, 1, (1, 1), layers[-1]))  # => 16 X 32 X 32
    layers.extend(make_basic_block(16, 16, 1, (1, 1), layers[-1]))
    # # Block 2
    layers.extend(make_basic_block(16, 32, 2, (1, 1), layers[-1]))  # 32 X 16 X 16
    layers.extend(make_basic_block(32, 32, 1, (1, 1), layers[-1]))
    # # Block 3
    layers.extend(make_basic_block(32, 64, 2, (1, 1), layers[-1]))  # 64 X 8 X 8
    layers.extend(make_basic_block(64, 64, 1, (1, 1), layers[-1]))
    # # Average pool
    layers.append(AvgPool2d((8, 8), 1, layers[-1]))  # 1 X 1 X 64 (H X W X C)
    layers.append(FlattenLayer(layers[-1]))
    layers.append(LinearLayer(64, num_classes, layers[-1]))
    # End of network. Find loss and reverse network.
    layers.append(CrossEntropyLoss(layers[-1], num_classes))
    bwd_layers = list(reversed(layers))
    for out_layer, in_layer in zip(bwd_layers[1:], bwd_layers[:-1]):
        layers.append(GradientLayer(out_layer, [in_layer], layers[-1]))

    return layers


if __name__ == "__main__":
    from poet.chipsets import *

    CHIPSET = MKR1000
    print("### Network ###")
    net = resnet18_cifar(1)

    param_count = 0
    totalEnergy = 0
    for layer in net:
        param_count += layer.param_count
        totalEnergy += layer.energy(M4F)
        print(layer)

    print("### Total number of parameters in network:", param_count)
    print("### Total energy (forward + backward) cost is:", totalEnergy)

    # print("### Profiles ###")
    # resource = get_net_costs(net, CHIPSET)
    # for _list in resource:
    #     print(_list, resource[_list])
