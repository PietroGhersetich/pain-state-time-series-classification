# Human Pain State Classification using Recurrent Neural Networks

This repository contains the deep learning solution developed by **NoNameTeam** for the **Artificial Neural Networks and Deep Learning (AN2DL)** First Challenge at **Politecnico di Milano** (A.Y. 2025/2026).

The [Report](.report/histopathology_classification_report.pdf) found in this repository contains more details on the work done by the Team.

---

## Overview

This project tackles the classification of human pain states from multivariate time series data. The dataset contains **661 subjects**, each represented by a sequence of **160 time steps**, with 38 features per step describing joint measurements, pain surveys, and categorical body characteristics. The goal is to classify each subject into one of three classes: **no pain**, **low pain**, and **high pain**, evaluated primarily using the **Weighted F1-Score**.

Our approach prioritizes **subject-wise validation**, **sliding-window sequence construction**, and **regularized recurrent neural networks**. After comparing RNN, GRU, and LSTM architectures, a two-layer **Bidirectional LSTM (Bi-LSTM)** provided the strongest validation performance by capturing temporal dependencies while controlling overfitting through regularization.

---

## Key Challenges

Initial exploratory data analysis (EDA) revealed three primary bottlenecks:

* **Strong Class Imbalance**: The training set is heavily skewed toward the *no pain* class, which accounts for more than three quarters of the subjects. This makes the weighted F1-score more informative than accuracy alone and can make minority-class recognition difficult.
* **Outliers & Uninformative Features**: Some joint measurements contain extreme values, while the `joint_30` feature is constant across the dataset. We removed the constant feature and reduced the impact of outliers before training.
* **Temporal Structure & Subject Leakage**: Each subject is represented by a sequence of measurements rather than an independent feature vector. Splitting individual time windows at random could leak information about the same subject into both training and validation sets, so validation was performed at subject level.

The class distribution in the full training set is summarized below:

| Class ID | Pain State | Occurrences | Percentage |
| --- | --- | --- | --- |
| **0** | **No pain** | 511 | 77.31% |
| **1** | **Low pain** | 94 | 14.22% |
| **2** | **High pain** | 56 | 8.47% |
| **Total** |  | **661** | **100.00%** |

---

## Data Preprocessing Pipeline

To convert the raw subject-level data into an input suitable for sequence modeling, we implemented a preprocessing pipeline centered on **feature cleaning and sliding-window sequencing**:

1. **Feature Inspection**: Inspected feature distributions and variance, identifying outliers and the constant `joint_30` feature, which was removed.
2. **Categorical Encoding**: Converted the categorical body characteristics (`n_legs`, `n_hands`, and `n_eyes`) into numerical values and encoded the target labels as class IDs.
3. **Outlier Handling & Scaling**: Clipped continuous joint and pain-survey measurements at the 99th percentile, then applied Min-Max scaling using statistics computed on the training split.
4. **Sliding-Window Sequencing**: Split each subject's time series into overlapping windows, padding the end of a sequence when needed. The final notebook uses a window length of `18` time steps and a stride of `6`.
5. **Subject-Level Prediction Aggregation**: At inference time, combined a subject's window-level predictions using a majority vote to produce one pain-state label per subject.

---

## Model Architecture

We explored several recurrent architectures to identify a model capable of capturing temporal dependencies in the sensor data:

- **Recurrent Neural Network (RNN)**:
  - Used as a baseline to evaluate the task with a standard recurrent architecture.
- **Gated Recurrent Unit (GRU)**:
  - Tested in unidirectional and bidirectional configurations. The bidirectional GRU established a strong baseline, with a validation weighted F1-score of `0.9347`.
- **Bidirectional Long Short-Term Memory (`Bi-LSTM`)**:
  - Selected as the final architecture after experiments with recurrent depth, hidden size, and regularization.
  - Uses **two bidirectional LSTM layers** with a hidden size of `512` and a three-class classification head.
  - Contains approximately **8.5 million parameters**, providing high representational capacity for learning temporal patterns in the measurements.

---

## Training, Regularization & Validation

To improve generalization to unseen subjects and reduce overfitting, we paired systematic model comparison with a regularized training procedure:

* **Subject-Wise Validation & Hyperparameter Tuning**: Used subject-level splits and K-Fold Shuffle Split validation to evaluate configurations on users not seen during training. The grid-search notebook records recurrent model experiments and hyperparameter evaluation.
* **Optimization**: Trained the final Bi-LSTM with AdamW, a learning rate of `1.25e-3`, and a batch size of `256`, for up to `1000` epochs with early stopping (`patience = 60`).
* **Regularization**: Combined dropout (`0.35`), L1 regularization (`1e-6`), and L2 weight decay (`1e-4`) to manage the overfitting risk associated with the model's size.
* **Loss Function**: Compared standard Cross-Entropy Loss with class-weighted Cross-Entropy and label smoothing. The final model used **standard Cross-Entropy Loss**, which performed better when combined with strong regularization.
* **Feature Engineering & Ensembling**: Experiments with velocity and acceleration features, time-series autocorrelation, and model ensembling did not provide a consistent improvement over the final single Bi-LSTM.

---

## Results

Performance comparison of the main architectures on the held-out, subject-wise validation set:

| Model Architecture | Validation Accuracy | Validation Precision | Validation Recall | Validation F1 (Weighted) |
| --- | --- | --- | --- | --- |
| **Baseline (Bi-GRU)** | 93.45% | 93.51% | 93.45% | 0.9347 |
| **Bi-LSTM (Weighted Loss)** | 94.22% | 94.25% | 94.22% | 0.9423 |
| **Final Model (Bi-LSTM)** | **98.12%** | **98.13%** | **98.12%** | **0.9811** |

> **Key Finding & Limitations**: The final two-layer Bi-LSTM achieved a validation weighted F1-score of **0.9811**. The experiments showed that strong regularization with standard Cross-Entropy Loss was more effective than explicit class weighting, which yielded a slightly lower weighted F1-score (approximately `0.976`). The model's capacity creates an overfitting risk, and majority-vote aggregation assumes one final pain state per subject even though pain can vary over time.

---

## Repository Structure

```text
.
├── notebooks/
│   ├── grid_search_rnn_models.ipynb                      # Model exploration and hyperparameter optimization
│   └── best_model_bilstm.ipynb                           # Final two-layer Bi-LSTM implementation
├── report/
│   └── pain_state_classification_report.pdf              # Technical report
├── LICENSE                                               # MIT License
└── README.md                                             # Project documentation

```

---

## Team: NoNameTeam

[Pietro Ghersetich](https://www.linkedin.com/in/pietro-ghersetich/)

[Alessandro Mattiazzi](https://www.linkedin.com/in/alessandro-mattiazzi-2b9a48298/)

[Matteo Mugnai](https://www.linkedin.com/in/matteomugnai/)

[Tommaso Tron](https://www.linkedin.com/in/tommaso-tron-964162359/)

*Politecnico di Milano — Artificial Neural Networks and Deep Learning (November 2025)*
