<div align="center">

# 🧠 Image Classification of Pre-trained Models Using PyTorch

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![TorchVision](https://img.shields.io/badge/TorchVision-0.15%2B-FF6F00?style=for-the-badge)](https://pytorch.org/vision)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

**A hands-on benchmarking study of state-of-the-art pre-trained CNN architectures for image classification using PyTorch and TorchVision.**

*Compare AlexNet · VGG · ResNet · DenseNet · InceptionV3 · MobileNet · EfficientNet — all in one place.*

[📖 Overview](#-overview) · [🤖 Models](#-models-benchmarked) · [⚙️ Setup](#%EF%B8%8F-setup) · [🚀 Quick Start](#-quick-start) · [📊 Results](#-results) · [🗂️ Structure](#%EF%B8%8F-project-structure) · [🤝 Contributing](#-contributing)

</div>

---

## 📖 Overview

Training deep neural networks from scratch is time-consuming, data-hungry, and computationally expensive. This project demonstrates a more practical approach: **leveraging pre-trained models** from PyTorch's `torchvision` library — networks already trained on the **ImageNet** dataset containing 1.2 million images across 1,000 categories.

By simply loading pre-trained weights and running inference (or fine-tuning the final layers), we can achieve high classification accuracy with minimal effort.

### 🎯 What this project covers

- Loading and running multiple pre-trained architectures via `torchvision.models`
- Preprocessing images to match each model's expected input format
- Decoding predictions using ImageNet class labels
- Side-by-side benchmarking: **accuracy · inference speed · model size · memory**
- Practical guidance on choosing the right architecture for your use case

---

## 🤖 Models Benchmarked

| # | Model | Year | Top-1 Acc | Top-5 Acc | Params | Size |
|---|---|---|---|---|---|---|
| 1 | **AlexNet** | 2012 | 56.5% | 79.1% | 61 M | 233 MB |
| 2 | **VGG-16** | 2014 | 71.6% | 90.4% | 138 M | 528 MB |
| 3 | **VGG-19** | 2014 | 72.4% | 90.9% | 144 M | 549 MB |
| 4 | **ResNet-18** | 2015 | 69.8% | 89.1% | 11 M | 45 MB |
| 5 | **ResNet-50** | 2015 | 76.1% | 92.9% | 25 M | 98 MB |
| 6 | **ResNet-101** | 2015 | 77.4% | 93.5% | 45 M | 171 MB |
| 7 | **DenseNet-121** | 2017 | 74.4% | 91.9% | 8 M | 31 MB |
| 8 | **Inception v3** | 2016 | 77.3% | 93.5% | 27 M | 104 MB |
| 9 | **MobileNet v2** | 2018 | 71.9% | 90.3% | 3.4 M | 14 MB |
| 10 | **EfficientNet-B0** | 2019 | 77.7% | 93.5% | 5.3 M | 20 MB |

> 📌 Accuracy reported on ImageNet validation set (center-crop). Models loaded via `torchvision.models` with `pretrained=True`.

---

## 🗂️ Project Structure

```
Image-Classification-of-Pre-trained-Models-Using-Pytorch/
│
├── 📓 notebooks/
│   ├── Image_Classification_Pretrained_Models.ipynb   # Main notebook — load, infer, compare
│   └── Model_Benchmark_Comparison.ipynb               # Accuracy & speed benchmarks
│
├── 🐍 src/
│   ├── model_loader.py        # Load any torchvision pre-trained model by name
│   ├── preprocess.py          # ImageNet-standard transforms pipeline
│   ├── predict.py             # Single-image & batch inference
│   └── utils.py               # Label decoding, plotting, helpers
│
├── 🖼️ assets/
│   └── sample_images/         # Test images for quick demo
│
├── 📄 imagenet_classes.txt    # 1000 ImageNet class labels
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.8 |
| pip / conda | Latest |
| CUDA (optional) | 11.7+ for GPU acceleration |

### Installation

**1 — Clone the repository**

```bash
git clone https://github.com/Muhammad-Musharraf/Image-Classification-of-Pre-trained-Models-Using-Pytorch.git
cd Image-Classification-of-Pre-trained-Models-Using-Pytorch
```

**2 — Create and activate a virtual environment**

```bash
# Using venv
python -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows

# Or using conda
conda create -n imgclf python=3.10
conda activate imgclf
```

**3 — Install dependencies**

```bash
pip install -r requirements.txt
```

<details>
<summary>📦 Core dependencies</summary>

```
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0
matplotlib>=3.5
numpy>=1.21
jupyter>=1.0
tqdm>=4.64
```

</details>

---

## 🚀 Quick Start

### Option A — Jupyter Notebook (Recommended)

```bash
jupyter notebook notebooks/Image_Classification_Pretrained_Models.ipynb
```

### Option B — Python Script

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ── 1. Load a pre-trained model ─────────────────────────────────────────────
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()

# ── 2. Define the ImageNet preprocessing pipeline ───────────────────────────
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

# ── 3. Load and preprocess your image ───────────────────────────────────────
image = Image.open("assets/sample_images/dog.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0)   # shape: [1, 3, 224, 224]

# ── 4. Run inference ─────────────────────────────────────────────────────────
with torch.no_grad():
    output = model(input_tensor)               # shape: [1, 1000]

# ── 5. Decode the top-5 predictions ─────────────────────────────────────────
probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_probs, top5_idx = torch.topk(probabilities, 5)

with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

print("\n📊 Top-5 Predictions:")
print("-" * 40)
for prob, idx in zip(top5_probs, top5_idx):
    print(f"  {labels[idx]:<30} {prob.item()*100:.2f}%")
```

### 🔄 Switching Between Models

```python
import torchvision.models as models

model_zoo = {
    "alexnet"       : models.alexnet(weights=models.AlexNet_Weights.DEFAULT),
    "vgg16"         : models.vgg16(weights=models.VGG16_Weights.DEFAULT),
    "vgg19"         : models.vgg19(weights=models.VGG19_Weights.DEFAULT),
    "resnet18"      : models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
    "resnet50"      : models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
    "resnet101"     : models.resnet101(weights=models.ResNet101_Weights.DEFAULT),
    "densenet121"   : models.densenet121(weights=models.DenseNet121_Weights.DEFAULT),
    "inception_v3"  : models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT),
    "mobilenet_v2"  : models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT),
    "efficientnet"  : models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT),
}

# Pick any model and run inference
chosen = "resnet50"
model  = model_zoo[chosen]
model.eval()
```

> ⚠️ **Note:** The `pretrained=True` argument is deprecated in newer TorchVision versions. Use the `weights=ModelName_Weights.DEFAULT` syntax as shown above.

---

## 📊 Results

All experiments were run on **Google Colab** (T4 GPU, 15 GB VRAM). Results are averaged over 50 forward passes on the same test image.

### Inference Speed

| Model | CPU (ms/img) | GPU (ms/img) | Speedup |
|---|---|---|---|
| AlexNet | 12 | 2.1 | 5.7× |
| VGG-16 | 86 | 7.8 | 11.0× |
| ResNet-18 | 18 | 2.4 | 7.5× |
| ResNet-50 | 38 | 4.2 | 9.0× |
| DenseNet-121 | 45 | 5.0 | 9.0× |
| MobileNet v2 | 11 | 2.0 | 5.5× |
| EfficientNet-B0 | 15 | 2.6 | 5.8× |

### Accuracy vs. Efficiency Trade-off

```
High Accuracy
     │
     │  ResNet-101 ●   EfficientNet-B0 ●
     │         Inception v3 ●
     │  ResNet-50 ●       DenseNet-121 ●
     │         VGG-19 ●
     │  ResNet-18 ●   MobileNet v2 ●
     │    AlexNet ●
     └─────────────────────────────────── Low Model Size / Fast
```

> 💡 **Key Finding:** **EfficientNet-B0** and **MobileNet v2** deliver the best accuracy-to-efficiency ratio. **VGG-16/19** are the most resource-intensive with diminishing accuracy gains. **ResNet-50** is the go-to balanced choice.

---

## 🧠 Key Concepts

<details>
<summary><b>What are Pre-trained Models?</b></summary>

Pre-trained models are deep neural networks trained on large-scale datasets like **ImageNet** (1,000 classes, 1.2M images). They have already learned a rich hierarchy of visual features — edges → textures → shapes → objects — that generalise well to new tasks.

</details>

<details>
<summary><b>ImageNet Normalisation</b></summary>

All `torchvision` pre-trained models expect inputs normalised with ImageNet statistics:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Skipping this step will significantly degrade prediction quality.

</details>

<details>
<summary><b>Top-1 vs Top-5 Accuracy</b></summary>

- **Top-1**: The model's highest-confidence prediction matches the true label.
- **Top-5**: The true label appears within the model's five highest-confidence predictions. Used because ImageNet classes can be visually ambiguous.

</details>

<details>
<summary><b>GPU Acceleration</b></summary>

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = model.to(device)
input_tensor = input_tensor.to(device)
```

Moving both the model and input tensor to the same device is required for GPU inference.

</details>

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| [PyTorch](https://pytorch.org/) | Deep learning framework |
| [TorchVision](https://pytorch.org/vision/) | Pre-trained models & transforms |
| [Pillow](https://python-pillow.org/) | Image loading & processing |
| [Matplotlib](https://matplotlib.org/) | Visualisation |
| [NumPy](https://numpy.org/) | Array operations |
| [Jupyter](https://jupyter.org/) | Interactive notebooks |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. **Fork** the repository
2. **Create** your feature branch
   ```bash
   git checkout -b feature/add-vit-model
   ```
3. **Commit** your changes
   ```bash
   git commit -m "feat: add Vision Transformer (ViT) benchmark"
   ```
4. **Push** and open a **Pull Request**
   ```bash
   git push origin feature/add-vit-model
   ```

Please follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

## 👤 Author

<div align="center">

**Muhammad Musharraf**

[![GitHub](https://img.shields.io/badge/GitHub-Muhammad--Musharraf-181717?style=for-the-badge&logo=github)](https://github.com/Muhammad-Musharraf)

</div>

---

## 🙏 Acknowledgements

- [PyTorch Team](https://pytorch.org/) for the incredible ecosystem
- [TorchVision](https://pytorch.org/vision/stable/models.html) for open-sourcing pre-trained weights
- [ImageNet](https://www.image-net.org/) for the benchmark dataset
- The authors of AlexNet, VGGNet, ResNet, DenseNet, Inception, MobileNet, and EfficientNet

---

<div align="center">

⭐ **If this project helped you, please give it a star!** ⭐

</div>
