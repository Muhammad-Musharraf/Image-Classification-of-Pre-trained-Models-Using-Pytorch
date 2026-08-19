# 🖼️ Image Classification Using Pre-trained Models in PyTorch

A comparative study of popular pre-trained CNN architectures for image classification using PyTorch and TorchVision.

---

## 📌 Overview

This project explores and benchmarks multiple state-of-the-art pre-trained models available through `torchvision` for the task of image classification. Each model is loaded with weights pre-trained on ImageNet and evaluated on input images to compare predictions, confidence scores, and inference performance.

---

## 🧠 Models Covered

| Model        | Architecture Type     | Top-1 Acc (ImageNet) |
|--------------|----------------------|----------------------|
| AlexNet      | Classic CNN           | ~56.5%               |
| VGG16        | Deep CNN              | ~71.6%               |
| ResNet50     | Residual Network      | ~76.1%               |
| InceptionV3  | Multi-scale CNN       | ~77.3%               |
| DenseNet121  | Densely Connected CNN | ~74.4%               |
| MobileNetV2  | Lightweight CNN       | ~71.9%               |
| EfficientNet | Compound Scaling CNN  | ~77.7%               |

---

## 📁 Project Structure

```
Image-Classification-of-Pre-trained-Models-Using-Pytorch/
│
├── images/                        # Sample input images for classification
├── models/                        # Model loading and inference scripts
├── image_classification.ipynb     # Main Jupyter Notebook (end-to-end pipeline)
├── predict.py                     # Standalone prediction script
├── utils.py                       # Helper functions (preprocessing, label mapping)
├── imagenet_classes.txt           # 1000 ImageNet class labels
├── requirements.txt               # Python dependencies
└── README.md
```

---

## ⚙️ Requirements

- Python 3.7+
- PyTorch ≥ 1.9
- TorchVision ≥ 0.10
- NumPy
- Pillow
- Matplotlib

Install all dependencies with:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch torchvision numpy pillow matplotlib
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Muhammad-Musharraf/Image-Classification-of-Pre-trained-Models-Using-Pytorch.git
cd Image-Classification-of-Pre-trained-Models-Using-Pytorch
```

### 2. Run the Jupyter Notebook

```bash
jupyter notebook image_classification.ipynb
```

### 3. Run Prediction via Script

```bash
python predict.py --image images/sample.jpg --model resnet50 --top_k 5
```

**Arguments:**

| Argument    | Description                                      | Default     |
|-------------|--------------------------------------------------|-------------|
| `--image`   | Path to the input image                          | Required    |
| `--model`   | Model name (`resnet50`, `vgg16`, `alexnet`, ...) | `resnet50`  |
| `--top_k`   | Number of top predictions to display             | `5`         |

---

## 🔍 How It Works

1. **Load a Pre-trained Model** — Models are fetched from `torchvision.models` with ImageNet weights.
2. **Preprocess the Image** — Input images are resized to 224×224 (or 299×299 for Inception), converted to tensors, and normalized using ImageNet statistics:
   - Mean: `[0.485, 0.456, 0.406]`
   - Std: `[0.229, 0.224, 0.225]`
3. **Run Inference** — The model outputs a 1000-class probability distribution.
4. **Display Results** — Top-K predictions are mapped to human-readable class labels and displayed with confidence scores.

---

## 📊 Example Output

```
Model: ResNet50
Image: dog.jpg

Top-5 Predictions:
  1. golden retriever      — 87.43%
  2. Labrador retriever    — 6.21%
  3. kuvasz                — 2.11%
  4. Great Pyrenees        — 1.04%
  5. clumber spaniel       — 0.89%
```

---

## 📈 Model Comparison

| Model        | Params   | Model Size | Avg Inference Time (CPU) |
|--------------|----------|------------|--------------------------|
| AlexNet      | 61M      | ~233 MB    | Fast                     |
| VGG16        | 138M     | ~528 MB    | Slow                     |
| ResNet50     | 25M      | ~98 MB     | Moderate                 |
| DenseNet121  | 8M       | ~32 MB     | Moderate                 |
| MobileNetV2  | 3.4M     | ~14 MB     | Very Fast                |
| EfficientNet | 5.3M     | ~20 MB     | Fast                     |

> MobileNetV2 and EfficientNet offer the best trade-off between accuracy and speed, making them ideal for resource-constrained environments.

---

## 🧪 Pre-processing Pipeline

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

---

## 📚 References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [TorchVision Models](https://pytorch.org/vision/stable/models.html)
- [ImageNet Large Scale Visual Recognition Challenge (ILSVRC)](https://www.image-net.org/)
- He et al., *Deep Residual Learning for Image Recognition* (ResNet)
- Simonyan & Zisserman, *Very Deep Convolutional Networks* (VGG)
- Tan & Le, *EfficientNet: Rethinking Model Scaling for CNNs*

---

## 👤 Author

**Muhammad Musharraf**
- GitHub: [@Muhammad-Musharraf](https://github.com/Muhammad-Musharraf)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
