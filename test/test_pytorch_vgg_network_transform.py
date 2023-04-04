from poet.power_computation import InputLayer
import torch
from poet.architectures.network_transform_pytorch import network_transform

# transforms VGG Model network layers into POET computation layers

batch_size = (1,)
input_shape = (3, 32, 32)
num_classes = 10
layers = [InputLayer((batch_size, *input_shape))]

# VGG16 model transformation - https://download.pytorch.org/models/vgg16-397923af.pth
final_layers = network_transform(
    torch.hub.load("pytorch/vision:v0.10.0", "vgg16", pretrained=True), layers, batch_size, num_classes, input_shape
)
print(final_layers)
assert len(final_layers) > 0
