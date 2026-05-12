# transfer-learning

A transfer learning project using various extensions.

## Project Structure 

### Root Directory

- /data: dataset files
- /src: Python scripts and modules.
- README.md: Providing an overview and documentation.

## Quick Start
1. Clone the repo and navigate into it:
```bash
git clone git@github.com:johnpeterleo/transfer-learning.git
cd transfer-learning
```

### Install Requirements
```bash
pip install -r requirements.txt
```

### How To Run 

1. Run extract.py

Parses image filenames and builds a pandas DataFrame with metadata (breed, label, file path, image index).
```bash
cd src
python extract.py
```

2. Run beginning.py 
        
Binary transfer learning experiment (Cat vs Dog classification) using the built in dataset from torchvision.datasets.OxfordIIITPet provided by torch, and not extract.py (which can be used for training on imbalanced data). This part uses Adam optimizer with 0.001 learning rate. The old final layer is replaced with "model.fc = nn.Linear(model.fc.in_features, 2)" which means that instead of ResNets 1000 or so outputs, we instead have two for Cat and Dog. Then the replaced final layer is fine-tuned with pet datasets training data.
```bash
cd src
python beginning.py
```
      
Produces these results during training:
```bash
Epoch 0/4
--------------------
train loss: 0.2561 acc: 0.9049
val loss: 0.0999 acc: 0.9747

Epoch 1/4
--------------------
train loss: 0.0794 acc: 0.9804
val loss: 0.0630 acc: 0.9847

Epoch 2/4
--------------------
train loss: 0.0701 acc: 0.9791
val loss: 0.0500 acc: 0.9877

Epoch 3/4
--------------------
train loss: 0.0482 acc: 0.9867
val loss: 0.0435 acc: 0.9880

Epoch 4/4
--------------------
train loss: 0.0519 acc: 0.9823
val loss: 0.0392 acc: 0.9872

Training done in 5m 50s
Best val acc: 0.9880
```

3. Run multi_class.py 
        
Multi-class transfer learning experiment for all 37 pet breeds using the built in dataset from torchvision.datasets.OxfordIIITPet provided by torch, and not extract.py (which can be used for training on imbalanced data). This part uses Adam optimizer with 0.001 learning rate. The replaced final layer (37 output instead of resnets own 1000 or so outputs) is fine-tuned with pet datasets training data.
```bash
cd src
python multi_class.py
```
        
Produces these results during training:
```bash
Epoch 0/4
--------------------
train loss: 1.8515 acc: 0.5870
val loss: 0.8111 acc: 0.8362

Epoch 1/4
--------------------
train loss: 0.6194 acc: 0.8728
val loss: 0.5301 acc: 0.8681

Epoch 2/4
--------------------
train loss: 0.4177 acc: 0.9084
val loss: 0.4757 acc: 0.8654

Epoch 3/4
--------------------
train loss: 0.3382 acc: 0.9209
val loss: 0.4193 acc: 0.8757

Epoch 4/4
--------------------
train loss: 0.2888 acc: 0.9291
val loss: 0.3824 acc: 0.8828

Training done in 5m 54s
Best val acc: 0.8828
```

## Contact
John Christensen - johnchristensen@outlook.com


Lidya Nasser -   


August Filannino -       


Samy Zouggari - 