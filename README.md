# Plant Disease Classification using Transfer Learning and Weighted Focal Loss

An end-to-end academic deep learning framework designed to automate the classification of crop health conditions across 38 distinct plant-disease pairings using transfer learning backbones and targeted robust loss adjustments.

---

## 📌 Problem Statement
Crop foliage pathologies present severe threats toward nutritional supply lines across modern agricultural operations. Manual structural analysis remains vulnerable to delays and subjective misclassifications. Automated deep computer vision systems frequently struggle with raw multi-class field datasets due to structural data imbalances—where standard plants distort visibility models against rare anomalies. 

This project implements a **Pre-trained MobileNetV2 architecture optimization process combined with a custom Weighted Focal Loss framework** to address class distribution skew, enabling accurate diagnostics across minority classes.

---

## 📊 Dataset Description
The model utilizes structural imagery compiled via the **New Plant Diseases Dataset (Kaggle Source)**.

* **Total Target Diagnostic Classes**: 38 unique categorical groupings (encompassing distinct plant species and explicit bacterial, fungal, or environmental disease variants).
* **Total Image Dataset Volume**: 96,644 distinct high-resolution color photographs.
* **Pipeline Splits Framework Data Partitioning Matrix**:
  
  | Dataset Split Partition | Image Count Dimensions | Allocation Ratio (%) |
  | :--- | :--- | :--- |
  | **Training Split** | 70,295 samples | 72.73% |
  | **Validation Split** | 17,572 samples | 18.18% |
  | **Testing Split** | 8,777 samples | 9.09% |

---

## 🏗️ Deep Learning Model Architecture
To optimize edge device flexibility, we use **MobileNetV2** as our feature extractor, combined with optimized classification layers.


### Structural Layer Sequences Breakdown
1. **Input Stage**: Standardized $(224 \times 224 \times 3)$ pixel resolution matrices.
2. **Deep Backbone Network**: MobileNetV2 base weights initialized via ImageNet training parameters. Backpropagation parameters are completely frozen to retain generic edge detection profiles.
3. **Global Average Pooling**: Compresses spatial activations to a 1D vector mapping, minimizing overfitting risks.
4. **Regularization Stage**: Dropout layer configured to an active rate of `0.30` to prevent co-adaptation of features.
5. **Output Top Classifier**: Dense transformation layer mapping features directly onto 38 discrete class categories using a Softmax activation function.

$$P(y = c \mid \mathbf{x}) = \frac{e^{\mathbf{z}_c}}{\sum_{j=1}^{38} e^{\mathbf{z}_j}}$$

---

## 🧪 Loss Function: Custom Weighted Focal Loss
Standard Categorical Cross-Entropy loss struggles when majority classes skew gradients away from underrepresented variants. This implementation replaces standard loss calculations with a custom **Weighted Focal Loss** formulation:

$$\text{WFL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

Where:
* $p_t$ represents the model's estimated probability for the correct ground-truth class.
* $\gamma$ (Gamma) acts as the focus adjustment tuning parameter (configured to `2.0`). It scales down the loss contribution from well-classified, easy examples, forcing updates to center on hard, misclassified samples.
* $\alpha_t$ (Alpha) represents the class weight inverse scaling factor, balancing structural dataset class representations.

---

## ⚙️ Training Configuration Parameters
* **Base Optimizer**: Adam (Initial Core Learning Rate parameter calibrated to $\eta = 10^{-4}$)
* **Batch Sizing Parameter Constraints**: 32 arrays per step
* **Training Epoch Bounds**: 20 structural calibration loops
* **Preprocessing Inversion Transformations**: Min-max normalization mapping raw pixel ranges into uniform standard scales bounded between $[-1, 1]$.

---

## 📈 Evaluation and Results Summary
The trained classifier achieves high diagnostic reliability across all test environments.

### Final Verification Performance Metrics
* **Final Target Validation Accuracy**: `~95.41%`
* **Macro General Precision Evaluation Metric**: `0.9538`
* **Macro Target Recall Evaluation Metric**: `0.9529`
* **Calculated Macro Balanced F1 Score Index**: `0.9533`

### Diagnostic Validation Visualizations
The performance plots below illustrate structural convergence patterns and feature-matching consistency across all target categories:

* **Training Convergence Matrix Curves**: Refer directly to `results/accuracy_curve.png` and `results/loss_curve.png`.
* **Structural Error Mapping Overlapping Profiles**: Look inside `results/confusion_matrix.png` to review class matching metrics across complex leaf mutations.

---

## 💻 How To Run

### 1. Installation and Dependencies Setup
Clone the workspace repository framework and install the required utility dependencies:
```bash
pip install -r requirements.txt
