# Efficient Chest X-ray Classification Through Neural Network Pruning

## Overview

This project explores how **neural network pruning** can **reduce the computational cost** of deep learning models for multi-label chest X-ray classification while **preserving predictive performance**.

Using a pretrained ResNet-34 model and the NIH ChestX-ray14 dataset, I investigated different pruning strategies and compared their effects on model size, computational complexity, and classification quality.

The main goal was to **understand the trade-offs** between model compression and predictive performance, particularly in a medical imaging task with imbalanced labels.

So how much can a deep learning model be compressed before its ability to identify radiological findings deteriorates significantly?


## Problem Statement

Deep learning models for medical image analysis can be computationally expensive. Reducing their size and computational requirements may make them easier to deploy in resource-constrained environments.

However, compression introduces a trade-off: a smaller model is not necessarily a better model, especially when some disease labels are much less frequent than others.

This project investigates that trade-off through controlled experiments with several structured pruning methods.


## Dataset

The experiments use the NIH ChestX-ray14 chest radiograph dataset, including a sample subset for initial experiments.

The task is multi-label classification: a single image can be associated with multiple radiological findings.

The model predicts 14 labels: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax

The dataset presents several challenges, including class imbalance and substantial variation in the frequency of individual findings.


## Methodology

### 1. Baseline model

I used a pretrained ResNet-34 architecture, adapted for 14 independent output labels.

The training pipeline includes:

- Image resizing and normalization

- Data augmentation during training

- A multi-label classification objective using BCEWithLogitsLoss

- Class weighting to account for imbalanced labels

- Separate training, validation, and test datasets

- Reproducibility controls through random seeds


### 2. Neural network pruning

I compared three pruning approaches:

**1. L1 Magnitude Pruning**

Removes less important channels based on weight magnitudes

**2. Network Slimming**

Uses batch normalization scale parameters to identify channels that can be removed

**3.Taylor Pruning**

Estimates channel importance using gradient-based Taylor approximations


After pruning, I **fine-tuned** the models to recover some of the lost predictive performance.

I also investigated different **pruning ratios** and **iterative pruning**, where compression is applied in multiple stages rather than all at once.



### 3. Evaluation

I evaluated the models using:

- **ROC-AUC** — ability to distinguish positive and negative labels

- **F1-score** — balance between precision and recall

- **Sensitivity** — proportion of positive cases detected

- **Parameter count** — model complexity

- **MACs** — approximate computational operations required for inference

- **Per-class ROC-AUC** — to examine how compression affects individual findings

I also analyzed performance changes across disease classes, paying particular attention to the **effects of pruning on less frequent labels**.


## Key findings

- **L1 pruning** reduced the **parameter count** by approximately **51%** and **MACs** by approximately **51%**, while maintaining a **ROC-AUC close to the baseline** in this experiment.

- **Network Slimming** achieved **the greatest reduction in model complexity**: approximately **76% fewer parameters** and **76% fewer MACs**. This came with a **larger decrease in ROC-AUC and F1-score**.

- **Taylor pruning** reduced model complexity by approximately **51%** and achieved **higher sensitivity** than the baseline, although its **ROC-AUC and F1-score were lower**.

- The methods **affected individual disease classes differently**. Per-class evaluation revealed that some findings experienced substantial performance degradation, while others showed improvements.

These results illustrate that **pruning cannot be evaluated by compression ratio alone**. The most appropriate method depends on the **balance between computational efficiency** and **the type of predictive errors** that matter for the task.


## Technical Stack

- Python
- PyTorch and torchvision
- torch-pruning
- scikit-learn
- NumPy and pandas
- Matplotlib
- Pillow
- KaggleHub
