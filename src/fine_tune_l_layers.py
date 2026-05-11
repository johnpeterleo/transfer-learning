import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# Fine-tune l layers simultaneously:
#
# - Load pretrained ResNet34
# - Freeze all pretrained layers
# - Unfreeze the last l layer groups simultaneously
# - Train only the unfrozen layers on the pet dataset
# - Compare validation accuracy for different values of l
#
# Example:
# l = 1 -> fc only
# l = 2 -> layer4 + fc
# l = 3 -> layer3 + layer4 + fc
#
# Goal:
# Investigate how the number of unfrozen pretrained layers
# affects transfer learning performance.