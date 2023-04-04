from poet.architectures.network_transform_tensorflow import network_transform
from poet.power_computation import InputLayer
import tensorflow as tf

# transforms VGG Model network layers into POET computation layers


# defining ResNet model in TensorFlow
def BasicBlock(inputs, num_channels, kernel_size, num_blocks, skip_blocks, name):
    """Basic residual block"""
    x = inputs
    for i in range(num_blocks):
        if i not in skip_blocks:
            x1 = ConvNormRelu(x, num_channels, kernel_size, strides=[1, 1], name=name + "." + str(i))
            x = tf.keras.layers.Add()([x, x1])
            x = tf.keras.layers.Activation("relu")(x)
    return x


def BasicBlockDown(inputs, num_channels, kernel_size, name):
    """Residual block with strided downsampling"""
    x = inputs
    x1 = ConvNormRelu(x, num_channels, kernel_size, strides=[2, 1], name=name + ".0")
    x = tf.keras.layers.Conv2D(
        num_channels, kernel_size=1, strides=2, padding="same", activation="linear", use_bias=False, name=name + ".0.downsample.0"
    )(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, name=name + ".0.downsample.1")(x)
    x = tf.keras.layers.Add()([x, x1])
    x = tf.keras.layers.Activation("relu")(x)
    return x


def ConvNormRelu(x, num_channels, kernel_size, strides, name):
    """Layer consisting of 2 consecutive batch normalizations with 1 first relu"""
    if strides[0] == 2:
        x = tf.keras.layers.ZeroPadding2D(padding=(1, 1), name=name + ".pad")(x)
        x = tf.keras.layers.Conv2D(
            num_channels, kernel_size, strides[0], padding="valid", activation="linear", use_bias=False, name=name + ".conv1"
        )(x)
    else:
        x = tf.keras.layers.Conv2D(
            num_channels, kernel_size, strides[0], padding="same", activation="linear", use_bias=False, name=name + ".conv1"
        )(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, name=name + ".bn1")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(
        num_channels, kernel_size, strides[1], padding="same", activation="linear", use_bias=False, name=name + ".conv2"
    )(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, name=name + ".bn2")(x)
    return x


def ResNet18(inputs):
    x = tf.keras.layers.ZeroPadding2D(padding=(3, 3), name="pad")(inputs)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding="valid", activation="linear", use_bias=False, name="conv1")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, name="bn1")(x)
    x = tf.keras.layers.Activation("relu", name="relu")(x)
    x = tf.keras.layers.ZeroPadding2D(padding=(1, 1), name="pad1")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="valid", name="maxpool")(x)
    x = BasicBlock(x, num_channels=64, kernel_size=3, num_blocks=2, skip_blocks=[], name="layer1")
    x = BasicBlockDown(x, num_channels=128, kernel_size=3, name="layer2")
    x = BasicBlock(x, num_channels=128, kernel_size=3, num_blocks=2, skip_blocks=[0], name="layer2")
    x = BasicBlockDown(x, num_channels=256, kernel_size=3, name="layer3")
    x = BasicBlock(x, num_channels=256, kernel_size=3, num_blocks=2, skip_blocks=[0], name="layer3")
    x = BasicBlockDown(x, num_channels=512, kernel_size=3, name="layer4")
    x = BasicBlock(x, num_channels=512, kernel_size=3, num_blocks=2, skip_blocks=[0], name="layer4")
    x = tf.keras.layers.GlobalAveragePooling2D(name="avgpool")(x)
    x = tf.keras.layers.Dense(units=1000, use_bias=True, activation="linear", name="fc")(x)
    return x


inputs = tf.keras.Input((None, None, 3))
resnet_tf = ResNet18(inputs)
model = tf.keras.Model(inputs, resnet_tf)
batch_size = (1,)
input_shape = (3, 32, 32)
num_classes = 10
layers = [InputLayer((batch_size, *input_shape))]

final_layers = network_transform([layer for layer in model.layers], layers, batch_size, num_classes, input_shape)
print(final_layers)
