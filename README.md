# Patient-Grouped Parkinson's Disease Screening Across Languages via LightGBM and Kolmogorov–Arnold Spline Ensembles

[![Paper PDF](https://img.shields.io/badge/IEEE%20JBHI-Paper%20PDF-red)](IEEE_Q1_Journal_Manuscript.pdf)
[![Supplementary Material](https://img.shields.io/badge/IEEE%20JBHI-Supplementary%20PDF-blue)](IEEE_Q1_Supplementary_Material.pdf)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official code, data processing pipelines, experimental benchmark results, and figure generation routines for the paper:
> **"Patient-Grouped Parkinson's Disease Screening Across Languages via LightGBM and Kolmogorov–Arnold Spline Ensembles"**  
> *Arunavo, Member, IEEE*  
> Prepared for IEEE Journal of Biomedical and Health Informatics (JBHI).

---

## 📌 Key Highlights & Methodological Contributions

1. **Strict Patient-Grouped Cross-Validation (`GroupKFold`):**
   - Eliminates speaker identity leakage from replicate recordings ($m_i \in [3, 8]$ per subject).
   - Quantifies that record-wise random splitting artificially inflates reported accuracy by **+6.56% on Turkish** ($90.90\%$ vs. $84.34\%$) and **+16.82% on English Oxford** ($93.64\%$ vs. $76.82\%$).

2. **Tree-Spline Ensemble (LightGBM + KAN):**
   - Combines orthogonal axis-aligned decision trees (LightGBM) with cubic B-spline Kolmogorov–Arnold Networks (KAN, $[D_{\text{in}}, 8, 2]$ with $G=5$ grid intervals).
   - Achieves recording-level accuracies of **85.08% on Turkish**, **93.49% on Bengali**, and **76.82% on Oxford** across five random seeds (`{42, 100, 2024, 777, 999}`).
   - Subject-level voting across replicate recordings raises screening accuracy to **86.27% on Turkish**, **100.00% on Bengali**, and **81.25% on Oxford**.

3. **Inner-Fold Youden's $J$ Threshold Calibration ($\tau^*$):**
   - Resolves the ~50% clinical specificity ceiling on 3:1 imbalanced cohorts, increasing Turkish specificity from **51.46% to 71.46%** without test leakage.

4. **Patient-Clustered Statistical Significance Testing:**
   - Accounts for intra-subject recording correlation using **Obuchowski's clustered $\chi^2_{\text{clus}}$ test** ($S=252, 120, 32$) and 25-fold multi-seed Wilcoxon signed-rank tests ($p = 0.0453$ against stronger LightGBM $\eta=0.10$).

5. **Cross-Lingual Acoustic Polarity Inversion & Few-Shot Learning Curves:**
   - Explains why static second cepstral coefficient (`MFCC_2`, $d = +0.95$ TR, $+0.96$ BN) and log-energy acceleration maintain consistent sign across languages, whereas first-order velocity (`Delta_MFCC_2`) flips sign ($d = -0.24$ in stationary vowel `/a/` vs. $d = +0.96$ in Bengali conversational transitions).
   - Demonstrates that cross-lingual transfer with Correlation Alignment (CORAL) provides a **+5.5% to +13.0% cold-start boost** when only 5% to 10% of target-language patients are labeled.

---

## 📂 Repository Structure

```text
├── figures/                               # Publication-ready high-resolution figures
│   ├── Fig1_System_Architecture.png       # 4-panel graphical system framework
│   ├── Fig2_MultiDataset_ROC_Curves.png   # 2x2 multi-cohort ROC curves & specificity recovery
│   ├── Fig3_Architecture_and_Ensemble_Ablation.png # 13-model benchmark & ensemble ablation
│   ├── Fig4_Leakage_and_Feature_Sensitivity.png    # Speaker leakage & feature sensitivity
│   ├── Fig5_SHAP_and_Polarity_Analysis.png         # TreeSHAP & cross-lingual polarity flip
│   ├── Fig6_KAN_Spline_Activations.png    # 6-panel cubic B-spline partial log-odds curves
│   ├── Fig7_CrossLingual_Learning_Curves.png # Few-shot target learning curves (0%-50%)
│   ├── FigS1_Confusion_Matrices.png       # Supplementary confusion matrices
│   └── FigS2_Training_Convergence.png     # Supplementary loss & ROC-AUC curves
│
├── results/                               # Empirical results & verified mappings
│   ├── revision_multiseed_ablation.json   # 5-seed metrics, ablations, & Obuchowski tests
│   ├── revision_learning_curves.json      # Few-shot learning curves & Cohen's d values
│   ├── homologous_feature_mapping.csv     # 27-feature Turkish-Bengali alignment table
│   └── verified_references.json           # All 45 references verified via CrossRef/DataCite
│
├── IEEE_Q1_Journal_Manuscript.pdf         # Compiled 7-page IEEE JBHI manuscript PDF
├── IEEE_Q1_Supplementary_Material.pdf     # Compiled 4-page Supplementary Material PDF
├── Q1_Journal_Manuscript.md               # Markdown version of manuscript
├── run_supervisor_revision_experiments.py # Core 5-seed benchmark & ablation pipeline
├── run_learning_curves_and_mfcc_analysis.py # Learning curves & MFCC polarity analysis
├── generate_revised_q1_figures.py         # Figure reproduction script (Figs. 1-7, S1-S2)
├── generate_ieee_q1_paper.py              # Full IEEE PDF & Supplementary compilation pipeline
└── README.md                              # This file
```

---

## 🚀 Quick Start & Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Arunavo-build/Perkinson-Disease.git
cd Perkinson-Disease
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install numpy scipy pandas scikit-learn lightgbm xgboost catboost torch matplotlib seaborn
```

---

## 📊 Reproducing Experiments & Figures

### Run the 5-Seed Multi-Cohort Benchmark & Ablations
```bash
python run_supervisor_revision_experiments.py
```

### Compute Cross-Lingual Learning Curves & Polarity Analysis
```bash
python run_learning_curves_and_mfcc_analysis.py
```

### Regenerate All Publication Figures
```bash
python generate_revised_q1_figures.py
```

### Recompile IEEE Manuscript & Supplementary Material PDFs
```bash
python generate_ieee_q1_paper.py
```

---

## 📝 Citation

If you use this codebase, models, or homologous feature mappings in your research, please cite:

```bibtex
@article{arunavo2026patientgrouped,
  author={Arunavo},
  journal={IEEE Journal of Biomedical and Health Informatics},
  title={Patient-Grouped Parkinson's Disease Screening Across Languages via LightGBM and Kolmogorov--Arnold Spline Ensembles},
  year={2026},
  volume={30},
  pages={1--7},
  doi={10.1109/JBHI.2026.xxxxxxx}
}
```

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
