<div align="center">

# 🌿 Plant Disease Classification
### Transfer Learning · Weighted Focal Loss · 38 Classes · 95.45% Accuracy

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![MobileNetV2](https://img.shields.io/badge/Backbone-MobileNetV2-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://arxiv.org/abs/1801.04381)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

**Automated plant disease diagnosis from leaf images — 38 disease categories, 14 crop species, production-ready inference pipeline.**

</div>

---

## 📋 Table of Contents

- [Project Snapshot](#-project-snapshot)
- [Key Results](#-key-results)
- [Highlights](#-highlights)
- [Why This Project Matters](#-why-this-project-matters)
- [What Makes This Interesting](#-what-makes-this-interesting)
- [Prediction Gallery](#-prediction-gallery)
- [Results & Training Curves](#-results--training-curves)
- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Loss Function](#-loss-function-weighted-focal-loss)
- [Training Configuration](#-training-configuration)
- [Classification Report](#-classification-report)
- [Model Information](#-model-information)
- [Repository Guide](#-repository-guide)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Engineering Achievements](#-engineering-achievements)
- [Future Work](#-future-work)
- [Conclusion](#-conclusion)
- [Team Contributions](#-team-contributions)
- [Contributing](#-contributing)
- [License](#-license)

---

## ⚡ Project Snapshot

| | |
|---|---|
| **Task** | Multi-Class Image Classification |
| **Backbone** | MobileNetV2 (ImageNet pretrained, frozen) |
| **Loss Function** | Custom Weighted Focal Loss |
| **Classes** | 38 disease / healthy categories |
| **Crops** | 14 plant species |
| **Dataset** | 87,777 images |
| **Test Accuracy** | **95.45%** |
| **Macro F1** | **95.42%** |
| **Framework** | TensorFlow / Keras |

---

## 🏆 Key Results

| Metric | Value |
|---|---|
| **Test Accuracy** | **95.45%** |
| **Macro F1** | **95.42%** |
| Macro Precision | 95.57% |
| Macro Recall | 95.42% |
| Weighted F1 | 95.46% |
| Test Set Size | 8,777 images |
| Classes | 38 |
| Dataset Size | 87,777 images |
| Backbone | MobileNetV2 |
| Loss Function | Weighted Focal Loss |

---

## ✅ Highlights

- ✅ **Transfer Learning** — MobileNetV2 frozen backbone + custom classification head
- ✅ **Custom Loss Function** — Weighted Focal Loss implemented from scratch in Keras
- ✅ **38-Class Classification** — single unified model across 14 crop species
- ✅ **Balanced Metrics** — macro F1 (95.42%) ≈ weighted F1 (95.46%), no majority-class bias
- ✅ **Full Evaluation Pipeline** — confusion matrix, classification report, training curves
- ✅ **Production-Ready Inference** — confidence scores, annotated prediction outputs
- ✅ **Mobile-Compatible Backbone** — deployable via TensorFlow Lite

---

## 🌱 Why This Project Matters

- 🌾 Plant diseases destroy **20–40% of global crop production** annually — early detection is a food security problem
- 📱 MobileNetV2's lightweight design enables **on-device, offline inference** for farmers in remote areas
- ⚖️ Standard cross-entropy fails on imbalanced disease datasets; **Weighted Focal Loss** directly addresses this
- 🔬 The model classifies **visually ambiguous disease pairs** that remain challenging due to strong morphological similarity
- 🚀 End-to-end pipeline is structured for extension to mobile apps, REST APIs, and edge IoT devices

---

## 🎯 What Makes This Interesting?

| Design Decision | Why It Matters |
|---|---|
| **MobileNetV2 backbone** | 3.4M params, 300M FLOPs — real-time inference, TFLite-deployable, 95%+ accuracy |
| **Weighted Focal Loss** | Down-weights easy examples; up-weights rare/hard classes — solves imbalance standard loss cannot |
| **Frozen feature extractor** | Converges in 20 epochs on standard hardware |
| **GlobalAveragePooling2D** | Spatial invariance — disease location within the frame does not affect prediction |
| **Embedded normalization** | Rescaling layer inside the model ensures correct preprocessing on export and deployment |

---

## 📸 Prediction Gallery

> Leaf image in → disease label + confidence score out.

| | |
|:---:|:---:|
| ![Apple Healthy](results/predictions/apple_healthy.png) | ![Apple Scab](results/predictions/apple_scab.png) |
| `Apple___healthy` · **99.96%** | `Apple___Apple_scab` · **99.74%** |
| ![Corn Healthy](results/predictions/corn_healthy.png) | ![Corn Common Rust](results/predictions/corn_rust.png) |
| `Corn_(maize)___healthy` · **100.00%** | `Corn_(maize)___Common_rust_` · **99.89%** |
| ![Grape Black Rot](results/predictions/grape_black_rot.png) | ![Potato Early Blight](results/predictions/potato_early_blight.png) |
| `Grape___Black_rot` · **98.95%** | `Potato___Early_blight` · **99.02%** |
| ![Potato Late Blight](results/predictions/potato_late_blight.png) | ![Tomato Healthy](results/predictions/tomato_healthy.png) |
| `Potato___Late_blight` · **99.74%** | `Tomato___healthy` · **98.93%** |
| ![Tomato Early Blight](results/predictions/tomato_early_blight.png) | ![Tomato Late Blight](results/predictions/tomato_late_blight.png) |
| `Tomato___Early_blight` · **53.74%** ⚠️ | `Tomato___Late_blight` · **67.91%** ⚠️ |

> ⚠️ Lower-confidence predictions are correct but uncertain — visually similar Tomato diseases share overlapping lesion morphology. These cases are flagged for agronomist review in a production setting.

---

## 📈 Results & Training Curves

| Accuracy | Loss |
|:---:|:---:|
| ![Accuracy Curve](results/accuracy_curve.png) | ![Loss Curve](results/loss_curve.png) |

Training and validation curves track closely across all 20 epochs — minimal overfitting. Accuracy climbs from ~71% (epoch 1) to ~95% (epoch 20). Loss descends steeply through epoch 10, then refines toward zero.

### Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

Strong diagonal concentration across all 38 classes. Residual confusion is confined to biologically similar Tomato disease pairs, confirming that the model's failure modes are semantically sensible.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/nidhihippargi/Plant_Disease_Classification.git
cd Plant_Disease_Classification

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### Train

```bash
python src/split_dataset.py --data_dir /path/to/PlantVillage
python src/train.py --epochs 20 --batch_size 32
# Outputs: plant_disease_model.keras, results/
```

### Predict

```bash
# Single image
python src/predict.py --model plant_disease_model.keras --image /path/to/leaf.jpg

# Batch
python src/predict.py --model plant_disease_model.keras --dir /path/to/images/
```

### Evaluate

```bash
python src/evaluate.py --model plant_disease_model.keras --test_dir /path/to/test/
# Outputs: results/classification_report.txt, results/confusion_matrix.png
```

---

## 🏗️ Architecture

```
Input (224 × 224 × 3)
   │
   ▼  Rescaling [−1, 1]          ← normalization embedded in model
   │
   ▼  MobileNetV2 (frozen)       ← ImageNet weights, 1280-dim feature output
   │
   ▼  GlobalAveragePooling2D     ← spatial invariance
   │
   ▼  Dropout (0.3)              ← regularization
   │
   ▼  Dense (38 units)           ← task-specific classification
   │
   ▼  Softmax
   │
   ▼  Prediction + Confidence Score
```

![Architecture Diagram](results/architecture_diagram.png)

~3.4M frozen backbone parameters + 38K trainable head parameters. Convergence in 20 epochs on a single GPU.

---

## 📊 Dataset

**New Plant Diseases Dataset (Augmented)** — 87,777 RGB leaf images, 38 categories.

| Split | Images |
|---|---|
| Training | 70,295 |
| Validation | 17,572 |
| Test | 8,777 |

<details>
<summary><b>Full Disease Category List (14 crops, 38 classes)</b></summary>

| Crop | Categories |
|---|---|
| Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| Blueberry | Healthy |
| Cherry | Powdery Mildew, Healthy |
| Corn (Maize) | Cercospora / Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape | Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis), Healthy |
| Orange | Haunglongbing (Citrus Greening) |
| Peach | Bacterial Spot, Healthy |
| Pepper Bell | Bacterial Spot, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Raspberry | Healthy |
| Soybean | Healthy |
| Squash | Powdery Mildew |
| Strawberry | Leaf Scorch, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

</details>

**Dataset challenges:** Cross-crop disease similarity (Potato vs. Tomato Early Blight), high intra-class variation across lighting and disease stages, and class imbalance (Tomato spans 10 sub-categories; Squash has one) — the core motivation for Weighted Focal Loss.

---

## ⚖️ Loss Function: Weighted Focal Loss

Standard cross-entropy over-represents majority classes and over-allocates gradient to already-correct easy examples — both problematic for this dataset.

**Focal Loss** (Lin et al., 2017):

```
FL(p_t) = −(1 − p_t)^γ · log(p_t)
```

When confidence is high (p_t → 1), gradient contribution is suppressed. When confidence is low (p_t → 0), full gradient is preserved — learning focuses on hard, misclassified examples.

**Per-class weights** (inversely proportional to frequency) additionally ensure minority classes like Squash and Raspberry receive proportionally stronger gradient signal.

**Outcome:** Squash Powdery Mildew → F1 = 1.000. Macro F1 (95.42%) ≈ Weighted F1 (95.46%) — no class is sacrificed for aggregate metrics.

---

## 🔧 Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss | Weighted Focal Loss |
| Batch Size | 32 |
| Epochs | 20 |
| Input Shape | 224 × 224 × 3 |
| Normalization | [−1, 1] |
| Dropout | 0.3 |
| Backbone | MobileNetV2 (frozen) |

**Augmentation:** horizontal flip, rotation, zoom, width/height shift, brightness adjustment.

---

## 📋 Classification Report

### Best Performing Classes

| Class | F1 Score |
|---|---|
| Squash Powdery Mildew | **1.0000** |
| Orange Citrus Greening | **0.9980** |
| Cherry Powdery Mildew | **0.9976** |
| Corn Common Rust | **0.9958** |
| Corn Healthy | **0.9957** |

Visually distinctive signatures (white powdery coating, brick-red pustules, asymmetric yellowing) leave no room for inter-class confusion.

### Most Challenging Classes

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Tomato Early Blight | 0.9016 | 0.7250 | **0.8037** |
| Tomato Septoria Leaf Spot | 0.7370 | 0.9128 | **0.8156** |
| Tomato Target Spot | 0.8151 | 0.8509 | **0.8326** |

All three produce irregular dark necrotic lesions on tomato foliage. Full differentiation requires spore morphology detail not recoverable from RGB images. F1 ≥ 0.804 on these classes remains practically useful; Grad-CAM is the logical next diagnostic step.

---

## 📦 Model Information

| Attribute | Value |
|---|---|
| **Format** | `.keras` (Keras native) |
| **Backbone** | MobileNetV2 (ImageNet pretrained) |
| **Input Resolution** | 224 × 224 × 3 |
| **Output Classes** | 38 |
| **Training Strategy** | Transfer Learning (frozen backbone) |
| **Loss Function** | Weighted Focal Loss |
| **Normalization** | Embedded Rescaling layer [−1, 1] |
| **Trainable Parameters** | ~38K (head only) |
| **Total Parameters** | ~3.4M |

---

## 🗂 Repository Guide

| Directory / File | Contents |
|---|---|
| `src/` | All training, inference, evaluation, and utility scripts |
| `results/` | Generated artifacts: curves, confusion matrix, classification report, predictions |
| `plant_disease_model.keras` | Saved trained model, ready for inference |
| `README.md` | Project documentation |

---

## 🌳 Project Structure

```
Plant_Disease_Classification/
│
├── src/
│   ├── train.py                    # Training loop + Weighted Focal Loss
│   ├── predict.py                  # Single-image and batch inference
│   ├── evaluate.py                 # Test set evaluation and artifact generation
│   ├── model.py                    # MobileNetV2 + classification head
│   ├── data_loader.py              # Dataset loading and class mapping
│   ├── data_pipeline.py            # tf.data pipeline: augmentation, batching, prefetch
│   ├── losses.py                   # Custom Weighted Focal Loss (Keras subclass)
│   ├── split_dataset.py            # Stratified train/val/test split
│   └── generate_architecture_diagram.py
│
├── results/
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│   ├── architecture_diagram.png
│   ├── classification_report.txt
│   └── predictions/
│
├── plant_disease_model.keras
└── README.md
```

---

## 🛠️ Tech Stack

| Technology | Role |
|---|---|
| Python 3.10+ | Core language |
| TensorFlow 2.x | Training, GPU acceleration, model export |
| Keras | Model definition, custom loss, serialization |
| NumPy | Array operations, metric computation |
| Matplotlib | Training curves, confusion matrix, prediction plots |
| Scikit-Learn | Classification report, confusion matrix utilities |
| Pillow | Image loading, resizing, format conversion |
| MobileNetV2 | Pretrained ImageNet backbone |

---

## 🎖 Engineering Achievements

- Built a 38-class plant disease classifier using MobileNetV2 transfer learning on 87,777 images
- Implemented Weighted Focal Loss from scratch as a custom Keras loss class to address class imbalance
- Achieved **95.45% test accuracy** and **95.42% macro F1** on 8,777 held-out images
- Designed a complete evaluation pipeline: confusion matrix, per-class classification report, training curves
- Developed a production-ready inference workflow producing confidence-scored, annotated prediction outputs
- Built a modular, extensible codebase structured for TFLite deployment, API integration, and further fine-tuning

---

## 🔭 Future Work

| Direction | Description |
|---|---|
| **Backbone fine-tuning** | Selectively unfreeze top MobileNetV2 blocks with differential learning rates for potential performance improvements |
| **Grad-CAM explainability** | Visualize which leaf regions drive predictions, particularly for low-confidence Tomato cases |
| **TFLite deployment** | Export to INT8 quantized TFLite for offline on-device inference |
| **FastAPI web service** | REST endpoint + Docker container for farm management software integration |
| **Real-time video inference** | Frame-level predictions from drone footage for continuous crop monitoring |
| **Field image robustness** | Augment with cluttered backgrounds, motion blur, perspective distortion |
| **Multi-label detection** | Handle co-infection scenarios where multiple diseases appear on one leaf |
| **Expanded crop coverage** | Add rice, wheat, cassava — crops with significant global food security relevance |

---

## ✅ Conclusion

This project delivers a complete plant disease classification pipeline: **95.45% test accuracy** and **macro F1 of 95.42%** across 38 categories, validated on 8,777 held-out images.

The key contributions are a custom **Weighted Focal Loss** that prevents majority-class dominance and a **MobileNetV2** transfer learning pipeline that converges in 20 epochs on standard hardware. Failure modes are confined to biologically ambiguous Tomato disease pairs — confirming the model learns meaningful visual features rather than dataset artifacts.

The codebase is modular, documented, and structured for the natural next steps: backbone fine-tuning, Grad-CAM analysis, and mobile deployment.

---

## 👥 Team Contributions

### [Tanisha Baslas](https://github.com/TanishaBaslas)
- Dataset preparation and organization
- Data preprocessing pipeline development
- Dataset cleaning and class distribution management
- Data augmentation configuration
- Input pipeline setup for model training

### [Nidhi Hippargi](https://github.com/nidhihippargi)
- MobileNetV2 transfer learning implementation
- Model architecture design and customization
- Training pipeline development
- Hyperparameter selection and experimentation
- Weighted Focal Loss integration
- Model training, optimization, and performance improvement
- GitHub repository management and project integration

### [Kunal Kapri](https://github.com/Kunal-Kapri)
- Model evaluation and testing
- Classification report generation
- Confusion matrix analysis
- Accuracy and loss visualization generation
- Prediction pipeline implementation
- Sample prediction generation and confidence score visualization
- Documentation support and result interpretation

## 🤝 Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request. For significant changes, include relevant test results or evaluation metrics.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## ⭐ Support

If this project helped you learn about deep learning, computer vision, or agricultural disease detection, consider giving the repository a ⭐.

[![GitHub Repo stars](https://img.shields.io/github/stars/nidhihippargi/Plant_Disease_Classification?style=social)](https://github.com/nidhihippargi/Plant_Disease_Classification)
---

<div align="center">

**Built with 🌿 for precision agriculture and food security**

</div>
