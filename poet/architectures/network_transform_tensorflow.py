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
from poet.power_computation_transformer import QueryKeyValueMatrix, QKTMatrix, QKTVMatrix
from torchvision.models.resnet import BasicBlock, Bottleneck
import tensorflow as tf


##transforms input model's network layers to output graph with POET layers for TensorFlow models
def network_transform(net, layers, batch_size, num_classes, input_shape):
    for module in net:
        if isinstance(module, tf.keras.layers.Dense):
            lin_layer = LinearLayer(module.units, module.units, layers[-1])
            act_layer = ReLULayer(lin_layer)
            layers.extend([lin_layer, act_layer])
        if isinstance(module, tf.keras.layers.Activation) and module._name == "relu":
            relu_layer = ReLULayer(layers[-1])
            layers.append(relu_layer)
        if isinstance(module, tf.keras.layers.Conv2D):
            if module.padding == "valid":
                padding = (0, 0)
            elif module.padding == "same":
                padding = (1, 1)
            conv_layer = Conv2dLayer(1, module.filters, module.kernel_size, module.strides[0], padding, layers[-1])
            layers.append(conv_layer)
        if isinstance(module, tf.keras.layers.BatchNormalization):
            layers.append(BatchNorm2d(layers[-1]))
        if isinstance(module, tf.keras.layers.MaxPool2D):
            layers.append(MaxPool2d(module.pool_size, module.strides[0], layers[-1]))
        if isinstance(module, tf.keras.layers.GlobalAveragePooling2D):
            if module.keepdims:
                layers.append(GlobalAvgPool(layers[-1]))
        if isinstance(module, tf.keras.layers.Activation) and module._name == "tanh":
            tanh_layer = TanHLayer(layers[-1])
            layers.append(tanh_layer)
        if isinstance(module, tf.keras.layers.Activation) and module._name == "sigmoid":
            sigmoid_layer = SigmoidLayer(layers[-1])
            layers.append(sigmoid_layer)
        if isinstance(module, tf.keras.layers.Flatten):
            flatten_layer = FlattenLayer(layers[-1])
            layers.append(flatten_layer)
        if isinstance(module, tf.keras.layers.Dropout):
            dropout_layer = DropoutLayer(layers[-1])
            layers.append(dropout_layer)
    return layers
