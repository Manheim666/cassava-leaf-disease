# Cassava Leaf Disease Classification

An end-to-end computer vision pipeline developed for the Kaggle **Cassava Leaf Disease Classification** competition hosted by Makerere University AI Lab, Uganda. 

This project fine-tunes modern deep learning vision architectures to classify phone photos of cassava leaves into four distinct disease categories or as healthy plant tissue.

---

## 📌 Highlights & Results

- **Validation Setup:** 5-fold Stratified K-Fold holdout split (Fold 0 held-out: 4,280 images).
- **Test-Time Augmentation (TTA):** 3-view evaluation (Original, Horizontal Flip, Vertical Flip).
- **Ensemble Result:** 50/50 probability averaging of ConvNeXt-Small and EfficientNetV2-S.

| Model | Resolution | Accuracy | Macro-F1 |
| :--- | :---: | :---: | :---: |
| **Model 1:** ConvNeXt-Small | $384 \times 384$ | 0.896 | 0.810 |
| **Model 2:** EfficientNetV2-S (with CutMix) | $448 \times 448$ | 0.896 | 0.808 |
| **Ensemble:** 50/50 Average | Dynamic | **0.899** | **0.816** |

### Per-Class F1 Score Breakdown

| Class | Label | ConvNeXt-Small | EfficientNetV2-S | Ensemble |
| :--- | :---: | :---: | :---: | :---: |
| Cassava Bacterial Blight (CBB) | 0 | 0.684 | 0.668 | **0.689** |
| Cassava Brown Streak Disease (CBSD) | 1 | 0.817 | 0.815 | **0.823** |
| Cassava Green Mottle (CGM) | 2 | 0.820 | 0.830 | **0.828** |
| Cassava Mosaic Disease (CMD) | 3 | 0.964 | 0.964 | **0.965** |
| Healthy | 4 | 0.764 | 0.763 | **0.775** |

---

## 📊 Dataset Overview

The dataset contains **21,397 labelled photos** taken in agricultural fields across Uganda using ordinary smartphone cameras under varying lighting conditions, angles, and distances.

| Class ID | Name | Count | Distribution |
| :---: | :--- | :---: | :---: |
| 0 | Cassava Bacterial Blight (CBB) | 1,087 | ~5% |
| 1 | Cassava Brown Streak Disease (CBSD) | 2,189 | ~10% |
| 2 | Cassava Green Mottle (CGM) | 2,386 | ~11% |
| 3 | Cassava Mosaic Disease (CMD) | 13,158 | ~61% |
| 4 | Healthy | 2,577 | ~12% |

### Domain Challenges
1. **Severe Class Imbalance:** CMD accounts for 61% of all samples. Accuracy alone can be misleading; thus, **Macro-F1** is used to monitor performance across minority classes.
2. **Label Noise:** An estimated 10–20% of training annotations contain noise or mislabeling. Regularization techniques like **Label Smoothing ($0.1$)** and **CutMix** are incorporated to improve model robustness.

---

## 🏗️ Architecture & Pipeline Overview

```
                      21,397 Farmer Phone Photos
                                 │
                     StratifiedKFold(n_splits=5)
                                 │
            ┌────────────────────┴────────────────────┐
            ▼                                         ▼
   17,117 Training Images                  4,280 Held-Out Images
            │                                         │
 ┌──────────┴──────────┐                   ┌──────────┴──────────┐
 │       Part A        │                   │       Part B        │
 │   ConvNeXt-Small    │                   │  EfficientNetV2-S   │
 │   @ 384x384 px      │                   │   @ 448x448 px      │
 │  (ImageNet-22k Pre) │                   │  (ImageNet-21k Pre) │
 └──────────┬──────────┘                   └──────────┬──────────┘
            │                                         │
   10 Epochs / OneCycle               10 Epochs / OneCycle + CutMix
            │                                         │
   Save Best Checkpoint                      Save Best Checkpoint
            │                                         │
    Predict Holdout (TTA)                     Predict Holdout (TTA)
            │                                         │
            └────────────────────┬────────────────────┘
                                 ▼
                     50/50 Probability Average
                                 │
                                 ▼
                    Final Metrics & Evaluation
```

---

## 🛠️ Project Structure

```text
.
├── Part_A_ConvNeXt.py       # Fine-tuning ConvNeXt-Small @ 384px
├── Part_B_EfficientNet.py   # Fine-tuning EfficientNetV2-S @ 448px with CutMix
├── requirements.txt         # Core dependencies
└── README.md                # Documentation
```

### Key Components

- **Pretrained Transfer Learning:** Employs models fine-tuned from ImageNet-22k and ImageNet-21k weights via `timm`.
- **Data Augmentation:** `RandomResizedCrop` (0.5–1.0 scale), `RandomHorizontalFlip`, `RandomVerticalFlip`, and `ColorJitter`.
- **Optimization:** `AdamW` optimizer ($\text{lr}=1\text{e-}4 \text{ to } 2\text{e-}4$, $\text{weight\_decay}=0.05$) combined with a `OneCycleLR` schedule.
- **Mixed Precision:** Accelerated training using `torch.autocast` and `GradScaler`.

---

## 🚀 How to Run

### Kaggle Execution (Recommended)
1. Open a new Kaggle notebook and attach the dataset: [Cassava Leaf Disease Classification](https://www.kaggle.com/c/cassava-leaf-disease-classification).
2. Enable GPU accelerator (e.g., NVIDIA P100 or T4) and set **Internet** to ON (to fetch `timm` pretrained weights).
3. Execute all cells. Full execution takes approximately **2.5 hours**.

### Local Execution Setup

1. **Install Dependencies:**
   ```bash
   pip install torch torchvision timm scikit-learn pandas numpy pillow
   ```

2. **Directory Structure:** Ensure dataset files are available:
   ```text
   /kaggle/input/cassava-leaf-disease-classification/
       ├── train.csv
       ├── label_num_to_disease_map.json
       └── train_images/
   ```

3. **Run Training:**
   Execute Part A followed by Part B to generate validation predictions and output metrics to `/kaggle/working/metrics.json`.

---

## 🔍 Key Insights & Analysis

1. **Why Ensemble?** Both ConvNeXt-Small and EfficientNetV2-S achieved identical individual accuracies ($0.896$). However, because they use distinct architectural paradigms (Transformer-inspired large $7\times 7$ kernels vs. fused inverted bottleneck convolutional blocks), they make non-overlapping errors. Simple $50/50$ averaging resolved $14$ additional misclassifications.
2. **Hardest Classes:** *Cassava Bacterial Blight (CBB)* ($F_1 \approx 0.689$) and *Healthy* ($F_1 \approx 0.775$) present the greatest challenge. Early CBB lesions are small and visually resemble undamaged leaves, whereas healthy samples are frequently misclassified across various disease classes.
3. **Solved Classes:** *Cassava Mosaic Disease (CMD)* achieves an $F_1$ score of $0.965$ due to its distinctive yellow-green mosaic patterns and abundant representation in the dataset.

---

## 📄 License
This project is open-source under the MIT License.
