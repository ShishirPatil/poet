import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.fx import symbolic_trace
import poet.architectures.remat_and_paging as remat_and_paging
import copy

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
num_epochs = 150
learning_rate = 0.01

# Image preprocessing modules
transform = transforms.Compose([transforms.Pad(4), transforms.RandomHorizontalFlip(), transforms.RandomCrop(32), transforms.ToTensor()])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root="../../data/", train=True, transform=transform, download=True)

test_dataset = torchvision.datasets.CIFAR10(root="../../data/", train=False, transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

# Load in mode
model = torchvision.models.resnet18(pretrained=True)
if torch.cuda.is_available():
    model.cuda()

# convert model to graph
traced = symbolic_trace(model)
torchcopy = copy.deepcopy(traced)

# layers to be rematerialized and paged
remat_layers = ["layer1.1.conv1", "layer1.0.bn1"]
paging_layers = ["layer2.1.conv1", "layer2.0.bn1"]

# intermediate storage of remat layers to be recomputed later
remat_list = []

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# Train the model
total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass - includes removing layer temporarily (remat) and paging out layer to cpu
        for n in torchcopy.graph.nodes:
            if n.target in remat_layers:
                model_indexer = remat_and_paging.get_all_parent_layers(model, n.target)[0]
                layer = getattr(model_indexer[0], model_indexer[1])
                layer.register_forward_hook(remat_and_paging.memory_saving(model_indexer, n, True, False, remat_list))
            # if n.target in paging_layers:
            #     model_indexer = remat_and_paging.get_all_parent_layers(model, n.target)[0]
            #     layer = getattr(model_indexer[0], model_indexer[1])
            #     layer.register_forward_hook(remat_and_paging.memory_saving(model_indexer, n, False, True, remat_list))
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        for n in torchcopy.graph.nodes:
            if n.target in remat_layers:
                model_indexer = remat_and_paging.get_all_parent_layers(model, n.target)[0]
                layer = getattr(model_indexer[0], model_indexer[1])
                layer.register_backward_hook(remat_and_paging.reuse_layer(model_indexer, n, True, False, remat_list))
            # if n.target in paging_layers:
            #     model_indexer = remat_and_paging.get_all_parent_layers(model, n.target)[0]
            #     layer = getattr(model_indexer[0], model_indexer[1])
            #     layer.register_backward_hook(remat_and_paging.reuse_layer(model_indexer, n, layer, False, True, remat_list))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print("Accuracy of the model on the train images: {} %".format(100 * correct / total))
    # Decay learning rate
    if (epoch + 1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        for n in torchcopy.graph.nodes:
            if n.target in remat_layers:
                model_indexer = remat_and_paging.get_all_parent_layers(model, n.target)[0]
                layer = getattr(model_indexer[0], model_indexer[1])
                layer.register_forward_hook(remat_and_paging.memory_saving(model_indexer, n, remat_list))
            # if n.target in paging_layers:
            #     model_indexer = remat_and_paging.get_all_parent_layers(model, n.target)[0]
            #     layer = getattr(model_indexer[0], model_indexer[1])
            #     layer.register_forward_hook(remat_and_paging.memory_saving(model_indexer, n, False, True, remat_list))
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print("Accuracy of the model on the test images: {} %".format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), "resnet.ckpt")
