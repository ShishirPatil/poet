from poet.power_computation import InputLayer
import torchvision.models
from poet.architectures.network_transform_pytorch import network_transform

# transforms ResNet Model network layers into POET computation layers

batch_size = (1,)
input_shape = (3, 32, 32)
num_classes = 10
layers = [InputLayer((batch_size, *input_shape))]

#Resnet18 model transformation - https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py, commit: 7dc5e5bd60b55eb4e6ea5c1265d6dc7b17d2e917
final_layers = network_transform(torchvision.models.resnet18(pretrained=True), layers, batch_size, num_classes, input_shape)
print(final_layers)

#Resnet50 model transformation - https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py, commit: 7dc5e5bd60b55eb4e6ea5c1265d6dc7b17d2e917
final_layers = network_transform(torchvision.models.resnet50(pretrained=True), layers, batch_size, num_classes, input_shape)
print(final_layers)
