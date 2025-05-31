# Sliding RandAugment - Manual & Setup Guide

This repository contains a configurable Python Jupyter notebook implementing **Sliding RandAugment**, a lightweight and effective data augmentation method designed for traffic sign classification. This README is focused on usage and configuration options for users running the notebook.

---

## 📁 File Structure

```
📦Sliding-RandAugment/
 ┣ 📜SlidingRandAugment.ipynb  ← Main notebook to configure, train, and evaluate
 ┣ 📁MODELS/                   ← Saved models
 ┣ 📁DATASETS/                 ← Training datasets (GTSRB, ETSD, CIFAR-10)
 ┣ 📁TrainingGTSRB/, etc.      ← Training logs
 ┣ 📜TEST_results.csv          ← Testing results
```

---

## ⚙️ Configuration Options

All configurable sections are found in the `SlidingRandAugment.ipynb` notebook.

### 1. ✅ Device Setup

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Optional: force CPU only
# device = torch.device('cpu')
```

---

### 2. 📦 Dataset Selection

Datasets must be pre-prepared or will be created automatically:

```python
# Uncomment datasets you want to use
GTSRB = GTSRB_LOADER("DATASETS/gtsrb/Train", transform=transform) #GTSRB
datasets.append(('GTSRB', GTSRB))

GTSRB_BALANCED = GTSRB_LOADER("DATASETS/gtsrb/Train_BALANCED", transform=transform) #GTSRB BALANCED
datasets.append(('GTSRB_BALANCED', GTSRB_BALANCED))

CIFAR10 = Cifar10("DATASETS/cifar10/train", transform=transform_cifar) #CIFAR-10
datasets.append(('CIFAR10', CIFAR10))

ETSD = ETSD_LOADER("DATASETS/etsd/Training", transform=transform) #ETSD
datasets.append(('ETSD', ETSD))

ETSD_BALANCED = ETSD_LOADER("DATASETS/etsd/Training_BALANCED", transform=transform) #ETSD BALANCED
datasets.append(('ETSD_BALANCED', ETSD_BALANCED))
```

---

### 3. 🔁 Augmentation Method

Select which augmentation to apply during training:

```python
orig_Randaugment = False
sliding_Randaugment = False
```
If either one is set to True, please edit training loop accordingly, to prevent duplicate running

If not set manually, all three will be run in sequence for comparison.

---

### 4. 🧠 Model Selection

```python
# Uncomment the models you want to train
MYmodels.append('DeepThin')
MYmodels.append('MobilNetV2')
MYmodels.append('SqueezeNet')
MYmodels.append('WideResNet')
```

---

### 5. 🔧 Training Parameters

```python
epochs = 17
batch_size = 32 
optimizer = 'SGD'        
learning_rate = 0.01
momentum = 0.9
weight_decay = 1e-4
```

---

### 6. 🔄 Sliding Window Settings

Customize how many augmentations are applied and how windows shift:

```python
# Number of augmentations to apply from each window
applied_simple_transforms = 2
applied_complex_transforms = 1

# Size of the sliding window
simple_window_size = 3
complex_window_size = 2
```

---

## 📈 Outputs

After training, you’ll find:

- **Saved models**: in the `MODELS/` folder as `.pth` files
- **Testing results**: written to `TEST_results.csv`
- **Training logs**: in the `TrainingGTSRB/, etc.` folders
- **Accuracy, Precision, Recall, F1 Score** for each model and augmentation setup

---

## 🛠 Requirements

Install with:

```bash
pip install torch torchvision albumentations timm numpy pandas matplotlib
```

Tested with:
- PyTorch 2.6.0+cu118
- Python 3.8+
- Works on both GPU (CUDA) and CPU

---

## 🧪 Notes

- All randomness is controlled via `torch.manual_seed(42)` for reproducibility.
- To run the full comparison between no augmentation, RandAugment, and Sliding RandAugment, leave augmentation selection intact.
- Adjust dataset paths if you want to use your own data.
