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
        
Binary transfer learning experiment (Cat vs Dog classification) using the built in dataset from torchvision.datasets.OxfordIIITPet provided by torch, and not extract.py (which can be used for training on imbalanced data)
```bash
cd src
python extract.py
```
      
Produces these results during training:
```bash
Epoch 0/4
--------------------
train loss: 0.3091 acc: 0.8772
val loss: 0.2331 acc: 0.9090

Epoch 1/4
--------------------
train loss: 0.1386 acc: 0.9478
val loss: 0.1427 acc: 0.9463

Epoch 2/4
--------------------
train loss: 0.0889 acc: 0.9677
val loss: 0.1502 acc: 0.9444

Epoch 3/4
--------------------
train loss: 0.0825 acc: 0.9701
val loss: 0.1853 acc: 0.9324

Epoch 4/4
--------------------
train loss: 0.0659 acc: 0.9755
val loss: 0.2486 acc: 0.9095

Training done in 15m 57s
Best val acc: 0.9463
```

3. Run multi_class.py 
        
Multi-class transfer learning experiment for all 37 pet breeds using the built in dataset from torchvision.datasets.OxfordIIITPet provided by torch, and not extract.py (which can be used for training on imbalanced data)
```bash
cd src
python multi_class.py
```
        
Produces these results during training:
```bash
Epoch 0/4
--------------------
train loss: 1.8490 acc: 0.4516
val loss: 2.0516 acc: 0.3949

Epoch 1/4
--------------------
train loss: 1.0916 acc: 0.6527
val loss: 1.8946 acc: 0.4623

Epoch 2/4
--------------------
train loss: 0.6957 acc: 0.7745
val loss: 1.8937 acc: 0.5119

Epoch 3/4
--------------------
train loss: 0.5742 acc: 0.8092
val loss: 1.6764 acc: 0.5737

Epoch 4/4
--------------------
train loss: 0.3918 acc: 0.8745
val loss: 1.8916 acc: 0.5435

Best val accuracy: 0.5737
```

## Contact
John Christensen - johnchristensen@outlook.com


Lidya Nasser -   


August Filannino -       


Samy Zouggari - 