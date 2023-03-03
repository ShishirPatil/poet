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
import torch.nn as nn
import torchvision.models
from torchvision.models.resnet import BasicBlock, Bottleneck


# transforms input model's network layers to output graph with POET layers for PyTorch models
def network_transform(net, layers, batch_size, num_classes, input_shape):
    if isinstance(net, torchvision.models.resnet.ResNet):
        modules = nn.Sequential(*list(net.children()))
    elif isinstance(net, torchvision.models.vgg.VGG):
        modules = list(net.children())
    else:
        modules = net
    for module in modules:
        if isinstance(module, nn.Sequential):
            sequential_modules = [child for child in module]
            input = network_transform(sequential_modules, [layers[-1]], batch_size, num_classes, input_shape)
            layers.extend(input[1:])
        if isinstance(module, BasicBlock) or isinstance(module, Bottleneck):
            input = network_transform(nn.Sequential(*list(module.children())), [layers[-1]], batch_size, num_classes, input_shape)
            layers.extend(input[1:])        
        if isinstance(module, nn.Linear):
            lin_layer = LinearLayer(module.in_features, module.out_features, layers[-1])
            act_layer = ReLULayer(lin_layer)
            layers.extend([lin_layer, act_layer])
        if isinstance(module, nn.ReLU):
            relu_layer = ReLULayer(layers[-1])
            layers.append(relu_layer)
        if isinstance(module, nn.Conv2d):
            conv_layer = Conv2dLayer(
                module.in_channels, module.out_channels, module.kernel_size, module.stride[0], module.padding, layers[-1]
            )
            layers.append(conv_layer)
        if isinstance(module, nn.BatchNorm2d):
            layers.append(BatchNorm2d(layers[-1]))
        if isinstance(module, nn.MaxPool2d):
            layers.append(MaxPool2d((module.kernel_size, module.kernel_size), module.stride, layers[-1]))
        if isinstance(module, nn.AvgPool2d):
            layers.append(AvgPool2d(module.kernel_size, module.stride, layers[-1]))
        if isinstance(module, nn.Tanh):
            tanh_layer = TanHLayer(layers[-1])
            layers.append(tanh_layer)
        if isinstance(module, nn.Sigmoid):
            sigmoid_layer = SigmoidLayer(layers[-1])
            layers.append(sigmoid_layer)
        if isinstance(module, nn.Flatten):
            flatten_layer = FlattenLayer(layers[-1])
            layers.append(flatten_layer)
        if isinstance(module, nn.Dropout):
            dropout_layer = DropoutLayer(layers[-1])
            layers.append(dropout_layer)
    return layers