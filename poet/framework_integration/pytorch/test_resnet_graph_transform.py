from torch.fx import symbolic_trace
import torchvision
from poet.architectures.graph_transformation import graph_transform

# transforms ResNet Model graph into POET layers nodes

# Resnet18 model transformation - https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py, commit: 7dc5e5bd60b55eb4e6ea5c1265d6dc7b17d2e917
traced = symbolic_trace(torchvision.models.resnet18(pretrained=True))
poet_traced = graph_transform(traced)
for n in poet_traced.graph.nodes:
    print(n.target)
    print(n.name)

# Resnet50 model transformation - https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py, commit: 7dc5e5bd60b55eb4e6ea5c1265d6dc7b17d2e917
traced = symbolic_trace(torchvision.models.resnet50(pretrained=True))
poet_traced = graph_transform(traced)
for n in poet_traced.graph.nodes:
    print(n.target)
    print(n.name)
