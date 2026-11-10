# Comprehensive Benchmark Results Across Three Multilingual Corpora

| Dataset | Model | Accuracy (Mean ± Std) | Sensitivity (Recall) | Specificity | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Turkish UCI | LightGBM | 83.47 ± 2.68% | 94.36% | 51.51% | 89.51% | 0.8690 |
| Turkish UCI | KAN | 81.86 ± 2.67% | 93.94% | 46.32% | 88.51% | 0.8027 |
| Turkish UCI | Naive Fusion | 83.58 ± 2.21% | 95.21% | 49.37% | 89.63% | 0.8665 |
| Turkish UCI | **Adaptive Fusion (Ours)** | **84.12 ± 2.08%** | **95.95%** | **49.33%** | **90.01%** | **0.8683** |
| Bengali BenSParX | LightGBM | 91.78 ± 0.82% | 93.56% | 90.00% | 91.93% | 0.9728 |
| Bengali BenSParX | KAN | 91.33 ± 2.10% | 88.67% | 94.00% | 91.11% | 0.9675 |
| Bengali BenSParX | Naive Fusion | 93.33 ± 2.02% | 92.00% | 94.67% | 93.26% | 0.9781 |
| Bengali BenSParX | **Adaptive Fusion (Ours)** | **92.56 ± 1.67%** | **92.89%** | **92.22%** | **92.59%** | **0.9772** |
| Oxford UCI | LightGBM | 73.32 ± 16.55% | 86.51% | 41.67% | 81.86% | 0.7395 |
| Oxford UCI | KAN | 75.64 ± 12.94% | 88.08% | 43.33% | 83.61% | 0.7523 |
| Oxford UCI | Naive Fusion | 76.24 ± 14.19% | 88.82% | 43.33% | 83.94% | 0.7309 |
| Oxford UCI | **Adaptive Fusion (Ours)** | **74.75 ± 14.98%** | **87.06%** | **43.33%** | **82.79%** | **0.7364** |