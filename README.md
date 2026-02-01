# Deep Learning Final Project - Van Gogh Classifier and Style Transfer

This repository contains my Deep Learning course final assignment, implemented in a single Jupyter notebook: **`DL_Project.ipynb`**.

The project has two parts:

- **Part A - Binary image classification (Transfer Learning):** classify whether a painting is **Vincent van Gogh** (`label=1`) or **not van Gogh** (`label=0`) using fine-tuned pretrained CNNs (**VGG19** and **AlexNet**).
- **Part B - Neural Style Transfer:** generate stylized images using optimization-based style transfer with 4 model variants (pretrained vs. fine-tuned VGG19/AlexNet) and evaluate the generated images using the fine-tuned classifiers.

## Notebook

- **`DL_Project.ipynb`** - end-to-end pipeline: data loading → training + Optuna tuning → evaluation → style transfer generation → style transfer evaluation.


## Part A - Classification (Transfer Learning)

### Models
- **VGG19 (ImageNet pretrained)**
- **AlexNet (ImageNet pretrained)**

### Training & evaluation
- Loss: **Cross Entropy**
- Optimizer: **Adam**
- Validation scheme: **4-fold Stratified K-Fold**
- Early stopping based on **validation loss**
- Metrics tracked per epoch:
  - loss, accuracy
  - **F1-score**
  - **AUC-ROC**
  - confusion matrix

### Hyperparameter optimization (Optuna)
The notebook runs an Optuna study per model (**15 trials**) to tune:
- learning rate (`1e-5` → `1e-3`, log-uniform)
- weight decay (`1e-6` → `1e-4`, log-uniform)
- early-stopping patience (3 → 10)
- batch size (64 → 128)

### Experiment tracking
The notebook logs trials to **Weights & Biases (wandb)** .

### Reported result (from the notebook output)
For the **best VGG model** that was loaded from disk, the notebook prints:

- train accuracy ≈ **0.9963**
- validation accuracy = **1.0**
- validation F1 = **1.0**
- validation AUC-ROC = **1.0**

(These values come from the saved `vgg_stats` that the notebook loads.)

---

## Part B - Neural Style Transfer

The notebook implements classical optimization-based neural style transfer:

- Extract intermediate features from `model.features`
- Compute **Gram matrices** for style features

### Models compared (4 variants)
1. VGG19 pretrained (not fine-tuned)
2. AlexNet pretrained (not fine-tuned)
3. VGG19 fine-tuned (from Part A)
4. AlexNet fine-tuned (from Part A)


### Style-transfer evaluation (classifier-based)
To compare the “Van Gogh-ness” of the generated images, the notebook:
- loads the generated images per model variant
- feeds them into the **fine-tuned VGG/AlexNet classifiers**
- computes the **average predicted probability** for label `1` (Van Gogh)

---
