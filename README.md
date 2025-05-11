# Sliding RandAugment
# Sliding RandAugment – Manual & Setup Guide

This repository contains a configurable Python Jupyter notebook implementing **Sliding RandAugment**, a lightweight and effective data augmentation method designed for traffic sign classification. This README is focused on usage and configuration options for users running the notebook.

---

## 📁 File Structure

```
📦Sliding-RandAugment/
 ┣ 📜SlidingRandAugment.ipynb  ← Main notebook to configure, train, and evaluate
 ┣ 📁MODELS/                   ← Saved model weights (.pth)
 ┣ 📁TrainingGTSRB/, etc.      ← Generated training datasets
 ┣ 📜results.csv               ← Logs for evaluation metrics
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
use_GTSRB = True
use_ETSD = True
use_CIFAR10 = True

# Toggle balanced dataset versions
balanced_GTSRB = True
balanced_ETSD = True
```

---

### 3. 🔁 Augmentation Method

Select which augmentation to apply during training:

```python
# 'none' - no augmentation
# 'rand' - original RandAugment
# 'sliding' - proposed Sliding RandAugment
augmentation_method = 'sliding'
```

If not set manually, all three will be run in sequence for comparison.

---

### 4. 🧠 Model Selection

```python
# Uncomment the models you want to train
use_deepthin = True
use_mobilenetv2 = True
use_squeezenet = False
use_wideresnet = False
```

---

### 5. 🔧 Training Parameters

```python
epochs = 17
batch_size = 64
optimizer = 'SGD'         # or 'Adam'
learning_rate = 0.03
use_scheduler = True
```

---

### 6. 🔄 Sliding Window Settings

Customize how many augmentations are applied and how windows shift:

```python
# Number of augmentations to apply from each pool
num_simple_aug = 2
num_complex_aug = 2

# Size of the sliding window
window_size = 4
window_step = 1
```

---

## 📈 Outputs

After training, you’ll find:

- **Saved model weights**: in the `MODELS/` folder as `.pth` files
- **Evaluation results**: written to `results.csv`
- **Training logs**: printed in notebook cells
- **Accuracy, Precision, Recall, F1 Score** for each model and augmentation setup

---

## 🛠 Requirements

Install with:

```bash
pip install torch torchvision albumentations timm numpy pandas matplotlib
```

Tested with:
- PyTorch 2.6.0
- Python 3.8+
- Works on both GPU (CUDA) and CPU

---

## 🧪 Notes

- All randomness is controlled via `torch.manual_seed(42)` for reproducibility.
- To run the full comparison between no augmentation, RandAugment, and Sliding RandAugment, leave `augmentation_method` unset.
- Adjust dataset paths if you want to use your own data.
