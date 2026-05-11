import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import time

# -----------------------
# Transform
# -----------------------
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# -----------------------
# Dataset (MULTI-CLASS)
# -----------------------
data_dir = ".."

train = datasets.OxfordIIITPet(
    root=data_dir,
    split="trainval",
    transform=data_transform,
    download=True,
    target_types="category"
)

test = datasets.OxfordIIITPet(
    root=data_dir,
    split="test",
    transform=data_transform,
    download=True,
    target_types="category"
)

dataloaders = {
    "train": DataLoader(train, batch_size=32, shuffle=True, num_workers=2),
    "val": DataLoader(test, batch_size=32, shuffle=False, num_workers=2)
}

dataset_sizes = {
    "train": len(train),
    "val": len(test)
}

# 37 breed classes
class_names = train.classes

# -----------------------
# Device
# -----------------------
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device}")

# -----------------------
# Visualize batch
# -----------------------
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

inputs, labels = next(iter(dataloaders["train"]))
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in labels])

# -----------------------
# Training loop
# -----------------------
def train_model(model, criterion, optimizer, scheduler=None, num_epochs=5):
    best_model_path = "best_multiclass_model.pt"
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs - 1}")
        print("-" * 20)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f"{phase} loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_path)

    print(f"\nBest val accuracy: {best_acc:.4f}")
    model.load_state_dict(torch.load(best_model_path))
    return model

# -----------------------
# Model (37 classes)
# -----------------------
model = models.resnet34(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 37)
model = model.to(device)

# -----------------------
# Loss + Optimizer
# -----------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------
# Train
# -----------------------
model = train_model(model, criterion, optimizer, scheduler=None, num_epochs=5)