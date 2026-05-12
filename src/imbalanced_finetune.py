import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score
import collections
import time


# Task1 — Fine-tuning with imbalanced classes.
# Same skeleton as fine_tune_l_layers.py, but instead of sweeping l = 1..4 we
# fix l = 2 and sweep three training strategies:
#   - baseline      : plain CE on the imbalanced subset (no correction)
#   - weighted_ce   : nn.CrossEntropyLoss(weight = inverse-frequency)
#   - oversampling  : WeightedRandomSampler with inverse-frequency weights
# Evaluation reports per-class accuracy + per-class / macro F1 (task asks for
# performance measures beyond overall accuracy).


# -----------------------
# Imbalance config
# -----------------------
CAT_BREEDS = {                              # 12 cat breeds (the minority group)
    "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair",
    "Egyptian Mau", "Maine Coon", "Persian", "Ragdoll",
    "Russian Blue", "Siamese", "Sphynx",
}
IMBALANCE_RATIO = 0.20                      # keep 20% of each cat-breed's images
L           = 2                             # fine-tune depth (best from l-sweep)
NUM_EPOCHS  = 10   # weighted CE needs ~2 extra epochs to catch up to baseline
LR          = 1e-3
BATCH_SIZE  = 64


# -----------------------
# Transform
# -----------------------
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# -----------------------
# Dataset
# -----------------------
data_dir = ".."

train = datasets.OxfordIIITPet(
    root=data_dir, split="trainval",
    transform=data_transform, download=True, target_types="category"
)
test = datasets.OxfordIIITPet(
    root=data_dir, split="test",
    transform=data_transform, download=True, target_types="category"
)

class_names = train.classes
num_classes = len(class_names)

# Resolve cat indices by name — torchvision sorts class names alphabetically,
# which interleaves cats and dogs, so a fixed range(12) doesn't work.
CAT_CLASS_INDICES = [i for i, n in enumerate(class_names) if n in CAT_BREEDS]
assert len(CAT_CLASS_INDICES) == 12


# -----------------------
# Build imbalanced subset: keep all dogs, 20% of each cat breed
# -----------------------
def make_imbalanced_subset(dataset, cat_indices, ratio=0.20, seed=42):
    rng = np.random.default_rng(seed)
    class_to_idxs = collections.defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_to_idxs[label].append(idx)

    selected = []
    for cls, idxs in class_to_idxs.items():
        if cls in cat_indices:
            n_keep = max(1, int(len(idxs) * ratio))
            chosen = rng.choice(idxs, size=n_keep, replace=False).tolist()
        else:
            chosen = idxs
        selected.extend(chosen)
    return Subset(dataset, selected)


imbal_train = make_imbalanced_subset(train, CAT_CLASS_INDICES, ratio=IMBALANCE_RATIO)
train_labels = [imbal_train.dataset[i][1] for i in imbal_train.indices]

dataset_sizes = {"train": len(imbal_train), "val": len(test)}


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
# Inverse-frequency weights (used by weighted_ce and oversampling)
# -----------------------
counts = np.bincount(train_labels, minlength=num_classes).astype(float)
counts_safe = np.where(counts == 0, 1, counts)

# normalised so total weight = num_classes (keeps loss scale comparable to CE)
ce_weights = 1.0 / counts_safe
ce_weights = ce_weights / ce_weights.sum() * num_classes
ce_weights = torch.tensor(ce_weights, dtype=torch.float).to(device)

sample_weights = torch.tensor(
    [1.0 / counts_safe[l] for l in train_labels], dtype=torch.float
)


# -----------------------
# Freeze all but last l blocks + fc (same helper as fine_tune_l_layers.py)
# -----------------------
def set_finetune_l_layers(model, l):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    layers = [model.layer4, model.layer3, model.layer2, model.layer1]
    for i in range(l):
        for param in layers[i].parameters():
            param.requires_grad = True


# -----------------------
# Training loop (same shape as fine_tune_l_layers.py)
# -----------------------
def train_model(model, criterion, optimizer, dataloaders, num_epochs=5, tag="model"):
    since = time.time()
    best_path = f"best_imbal_{tag}.pt"
    best_acc = 0.0
    torch.save(model.state_dict(), best_path)

    train_acc_list = []
    val_acc_list = []

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
            if phase == "train":
                train_acc_list.append(epoch_acc)
            else:
                val_acc_list.append(epoch_acc)
            print(f"{phase} loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_path)

    elapsed = time.time() - since
    print(f"\nTraining done in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"Best val acc: {best_acc:.4f}")

    model.load_state_dict(torch.load(best_path))
    return model, train_acc_list, val_acc_list


# -----------------------
# Per-class evaluation (overall acc + per-class acc + per-class/macro F1)
# -----------------------
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            preds = model(inputs.to(device)).argmax(1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    per_class_acc = np.array([
        (all_preds[all_labels == c] == c).mean() if (all_labels == c).any() else float("nan")
        for c in range(num_classes)
    ])
    f1_per_class = f1_score(all_labels, all_preds, average=None,
                            labels=list(range(num_classes)), zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report = classification_report(all_labels, all_preds,
                                   target_names=class_names, zero_division=0)
    return {
        "overall_acc":  (all_preds == all_labels).mean(),
        "per_class_acc": per_class_acc,
        "f1_per_class": f1_per_class,
        "f1_macro":     f1_macro,
        "report":       report,
    }


# -----------------------
# Fresh ResNet-34 per strategy (so runs don't share weights)
# -----------------------
def build_model():
    model = models.resnet34(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    set_finetune_l_layers(model, L)
    return model


# -----------------------
# Sweep over strategies — mirrors fine_tune_l_layers.py's l-sweep
# -----------------------
val_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

strategies = ["baseline", "weighted_ce", "oversampling"]
all_results = {}

for strat in strategies:
    print(f"\n{'='*30}\nStrategy: {strat}\n{'='*30}")

    if strat == "baseline":
        train_loader = DataLoader(imbal_train, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=0)
        criterion = nn.CrossEntropyLoss()
    elif strat == "weighted_ce":
        train_loader = DataLoader(imbal_train, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=0)
        criterion = nn.CrossEntropyLoss(weight=ce_weights)
    elif strat == "oversampling":
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(imbal_train, batch_size=BATCH_SIZE,
                                  sampler=sampler, num_workers=0)
        criterion = nn.CrossEntropyLoss()

    dataloaders = {"train": train_loader, "val": val_loader}

    model = build_model()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)

    model, train_acc, val_acc = train_model(
        model, criterion, optimizer, dataloaders,
        num_epochs=NUM_EPOCHS, tag=strat,
    )

    eval_results = evaluate(model, val_loader)
    print(f"\nOverall acc: {eval_results['overall_acc']:.4f}  "
          f"macro-F1: {eval_results['f1_macro']:.4f}")
    print(f"\nClassification report — {strat}")
    print(eval_results["report"])

    all_results[strat] = {
        "train_acc":     train_acc,
        "val_acc":       val_acc,
        "f1_per_class":  eval_results["f1_per_class"],
        "per_class_acc": eval_results["per_class_acc"],
        "f1_macro":      eval_results["f1_macro"],
        "overall_acc":   eval_results["overall_acc"],
    }


# -----------------------
# Plot 1: train/val accuracy curves per strategy
# -----------------------
plt.figure()
for strat, res in all_results.items():
    ep = range(1, len(res["val_acc"]) + 1)
    plt.plot(ep, res["train_acc"], "--", label=f"train {strat}")
    plt.plot(ep, res["val_acc"], label=f"val {strat}")
plt.title("Imbalance mitigation: train/val accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.savefig("compare_imbalance_strategies.png")
plt.close()


# -----------------------
# Plot 2: per-class F1 by strategy (cat columns marked)
# -----------------------
x = np.arange(num_classes)
width = 0.27
fig, ax = plt.subplots(figsize=(16, 5))
for i, strat in enumerate(strategies):
    ax.bar(x + i * width, all_results[strat]["f1_per_class"], width, label=strat)
for c in CAT_CLASS_INDICES:
    ax.axvline(c + width, color="navy", linewidth=0.6, alpha=0.4)
ax.set_xticks(x + width)
ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("F1")
ax.set_ylim(0, 1)
ax.set_title("Per-class F1 by strategy  (navy lines = cat breeds, reduced to 20%)")
ax.legend()
plt.tight_layout()
plt.savefig("per_class_f1.png")
plt.close()
