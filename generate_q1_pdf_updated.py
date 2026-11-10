from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        pass
    def footer(self):
        self.set_y(-15)
        self.set_font("Times", "I", 10)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def generate_q1_roadmap_pdf():
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_text_color(0, 0, 0)
    
    lh = 7.0

    # ===================== TITLE =====================
    pdf.set_font("Times", "B", 14)
    pdf.cell(0, 8, "Comprehensive Roadmap: Q1 Journal Publication Strategy", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Times", "I", 11)
    pdf.cell(0, 7, "Parkinson's Disease Detection via Adaptive Tree-Spline Fusion of LightGBM & KAN", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)

    # ===================== INTRO =====================
    pdf.set_font("Times", "", 12)
    intro = (
        "Respected Sir, after extensive benchmarking and iterative refinement based on your feedback, "
        "we have established a rigorous research methodology. This document presents our finalized "
        "8-phase roadmap designed to produce a publication-ready manuscript for a Q1-tier medical "
        "informatics journal (e.g., Computers in Biology and Medicine, Expert Systems with Applications, "
        "Artificial Intelligence in Medicine). Each phase addresses a specific reviewer expectation."
    )
    pdf.multi_cell(0, lh, intro, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # ===================== CURRENT VERIFIED RESULTS =====================
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, lh, "Verified Results So Far (Patient-wise GroupKFold, Zero Data Leakage)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Times", "", 11)
    pdf.ln(2)

    # Turkish Dataset Table
    pdf.set_font("Times", "B", 11)
    pdf.cell(0, lh, "Turkish UCI Dataset (756 samples, 252 patients, 754 features):", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Times", "", 11)
    
    col_w = [25, 55, 40]
    pdf.set_font("Times", "B", 10)
    pdf.cell(col_w[0], 6, "Rank", border=1, align="C")
    pdf.cell(col_w[1], 6, "Model", border=1, align="C")
    pdf.cell(col_w[2], 6, "Accuracy", border=1, align="C")
    pdf.ln()
    pdf.set_font("Times", "", 10)
    turkish_rows = [
        ("1", "Adaptive Fusion (Ours)", "85.18%"),
        ("2", "LightGBM", "84.26%"),
        ("3", "CatBoost", "82.54%"),
        ("4", "XGBoost", "82.01%"),
        ("5", "KAN", "81.86%"),
        ("6", "SAINT", "80.41%"),
        ("7", "NODE", "79.88%"),
        ("8", "GATE", "79.87%"),
        ("9", "TabTransformer", "75.91%"),
        ("10", "TabNet", "71.96%"),
    ]
    for rank, model, acc in turkish_rows:
        if rank == "1":
            pdf.set_font("Times", "B", 10)
        else:
            pdf.set_font("Times", "", 10)
        pdf.cell(col_w[0], 6, rank, border=1, align="C")
        pdf.cell(col_w[1], 6, model, border=1, align="C")
        pdf.cell(col_w[2], 6, acc, border=1, align="C")
        pdf.ln()
    pdf.ln(3)

    # Bengali Dataset Table
    pdf.set_font("Times", "B", 11)
    pdf.cell(0, lh, "Bengali BenSParX Dataset (900 samples, 120 patients, 71 features):", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Times", "", 11)

    pdf.set_font("Times", "B", 10)
    col_w2 = [25, 55, 40]
    pdf.cell(col_w2[0], 6, "Rank", border=1, align="C")
    pdf.cell(col_w2[1], 6, "Model", border=1, align="C")
    pdf.cell(col_w2[2], 6, "Accuracy", border=1, align="C")
    pdf.ln()
    pdf.set_font("Times", "", 10)
    bengali_rows = [
        ("1", "Adaptive Fusion (Ours)", "93.56%"),
        ("2", "LightGBM", "92.56%"),
        ("3", "KAN", "91.33%"),
    ]
    for rank, model, acc in bengali_rows:
        if rank == "1":
            pdf.set_font("Times", "B", 10)
        else:
            pdf.set_font("Times", "", 10)
        pdf.cell(col_w2[0], 6, rank, border=1, align="C")
        pdf.cell(col_w2[1], 6, model, border=1, align="C")
        pdf.cell(col_w2[2], 6, acc, border=1, align="C")
        pdf.ln()
    pdf.ln(5)

    # ===================== PHASE 1 =====================
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, lh, "Phase 1: Feature Selection & Deep Statistical Analysis", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Times", "", 12)
    p1_text = (
        "- The Turkish dataset contains 754 features, many of which act as noise and degrade deep learning performance. "
        "We will perform robust Feature Selection (ANOVA F-test, Mutual Information, and SHAP-based selection) to isolate "
        "the top most clinically significant features.\n"
        "- We will conduct rigorous statistical validations (Welch's t-tests, Mann-Whitney U tests) and generate "
        "correlation matrices to demonstrate a profound understanding of the dataset's acoustic properties.\n"
        "- Goal: Identify the minimal optimal feature subset that maximizes clinical discriminability."
    )
    pdf.multi_cell(0, lh, p1_text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # ===================== PHASE 2 =====================
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, lh, "Phase 2: True Clinical Validation via Patient-Isolated Splitting", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Times", "", 12)
    p2_text = (
        "- Many existing papers suffer from 'Data Leakage' by randomly splitting recordings, "
        "artificially inflating accuracy to 90%+ on the UCI dataset.\n"
        "- We permanently address this critical flaw by utilizing strict Patient-wise Group K-Fold "
        "Cross Validation across ALL testing, ensuring no patient's recordings overlap between "
        "training and testing sets.\n"
        "- This directly addresses the most common criticism from Q1 reviewers in this domain and "
        "proves true real-world clinical generalizability."
    )
    pdf.multi_cell(0, lh, p2_text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # ===================== PHASE 3 =====================
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, lh, "Phase 3: Adaptive Tree-Spline Weighted Fusion (Core Architectural Novelty)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Times", "", 12)
    p3_text = (
        "- We propose a novel hybrid architecture fusing LightGBM (gradient-boosted decision trees, "
        "excelling at discrete/tabular patterns) with KAN (Kolmogorov-Arnold Networks, excelling "
        "at learning continuous nonlinear spline functions).\n"
        "- Instead of naive 50-50 Soft Voting, we introduce Adaptive Weighted Fusion where the "
        "optimal weight ratio dynamically adapts to each dataset's feature dimensionality.\n"
        "- Verified Result: On the 754-feature Turkish dataset, a 60-40 (LightGBM-KAN) ratio achieves "
        "85.18%, beating standalone LightGBM (84.26%). On the 71-feature Bengali dataset, a natural "
        "50-50 ratio achieves 93.56%, beating standalone LightGBM (92.56%).\n"
        "- This 'Tree-Spline' complementarity is a strong novelty claim: trees capture discrete splits "
        "while splines capture smooth nonlinear vocal cord dynamics."
    )
    pdf.multi_cell(0, lh, p3_text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # ===================== PHASE 4 =====================
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, lh, "Phase 4: Explainable AI & Clinical Interpretability (XAI)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Times", "", 12)
    p4_text = (
        "- Medical journals strictly require models to be clinically transparent and interpretable.\n"
        "- We will integrate SHAP (SHapley Additive exPlanations) to visualize exactly which voice "
        "features (e.g., Jitter, Shimmer, HNR, MFCC) are driving predictions.\n"
        "- We will extract the symbolic mathematical splines learned by the KAN model and map them to "
        "physical vocal cord impairments observed in Parkinson's patients, effectively removing the "
        "'black box' nature of deep learning.\n"
        "- This dual-explainability (SHAP for LightGBM + Symbolic Splines for KAN) is a unique "
        "contribution not found in existing Parkinson's detection literature."
    )
    pdf.multi_cell(0, lh, p4_text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # ===================== PHASE 5 =====================
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, lh, "Phase 5: Global Multi-Dataset Consolidation & Cross-Lingual Validation", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Times", "", 12)
    p5_text = (
        "- To maximize the paper's scientific impact, we will not limit validation to just two datasets.\n"
        "- We will perform rigorous testing on multiple available global datasets (e.g., Turkish UCI, "
        "Bengali BenSParX, PC-GITA Spanish, Oxford Parkinson's, Italian PD, etc.).\n"
        "- Two types of validation will be performed:\n"
        "  (a) Within-Dataset: Patient-wise GroupKFold on each dataset individually.\n"
        "  (b) Cross-Dataset: Train on Dataset A, Test on Dataset B (and vice versa) to prove "
        "language-independent and recording-environment-independent robustness.\n"
        "- This multi-lingual, multi-corpus consolidation is extremely rare in PD detection literature "
        "and will be a powerful differentiator for Q1 acceptance."
    )
    pdf.multi_cell(0, lh, p5_text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # ===================== PHASE 6 (NEW) =====================
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, lh, "Phase 6: Ablation Study (Reviewer-Demanded)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Times", "", 12)
    p6_text = (
        "- Q1 reviewers almost always demand an Ablation Study to prove each component's contribution.\n"
        "- We will systematically measure the impact of removing each component:\n"
        "  (a) Fusion without Feature Selection vs. with Feature Selection.\n"
        "  (b) Fusion without Adaptive Weights (naive 50-50) vs. with Adaptive Weights.\n"
        "  (c) LightGBM alone vs. KAN alone vs. Fusion.\n"
        "  (d) Random Split vs. Patient-wise GroupKFold (to quantify the leakage inflation).\n"
        "- This table will conclusively prove that every component of our framework adds measurable value."
    )
    pdf.multi_cell(0, lh, p6_text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # ===================== PHASE 7 (NEW) =====================
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, lh, "Phase 7: Statistical Significance Testing", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Times", "", 12)
    p7_text = (
        "- Simply showing higher accuracy is not enough for Q1 journals. We must prove the "
        "improvement is statistically significant, not due to random chance.\n"
        "- We will perform McNemar's test (comparing prediction disagreements between Fusion vs. "
        "each baseline) and Wilcoxon signed-rank test (comparing fold-wise accuracy distributions).\n"
        "- We will report p-values and confidence intervals alongside all accuracy claims.\n"
        "- This statistical rigor is what separates a Q1-accepted paper from a Q2/Q3 rejection."
    )
    pdf.multi_cell(0, lh, p7_text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # ===================== PHASE 8 (NEW) =====================
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, lh, "Phase 8: Comprehensive Paper Drafting & Presentation", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Times", "", 12)
    p8_text = (
        "- Produce a polished proposed architecture diagram (block diagram) showing the complete "
        "pipeline: Raw Audio -> Feature Extraction -> Feature Selection -> Parallel LightGBM & KAN "
        "training -> Adaptive Weighted Fusion -> Final Prediction + Explainability.\n"
        "- Include comprehensive comparison tables against state-of-the-art methods from recent "
        "literature (2022-2025) on the same datasets, ensuring our method is positioned favorably.\n"
        "- Prepare high-quality figures: ROC curves, Confusion Matrices, SHAP beeswarm plots, "
        "KAN symbolic spline visualizations, and training convergence curves.\n"
        "- Draft the manuscript following the target journal's template with all required sections: "
        "Introduction, Related Work, Methodology, Experiments, Results, Discussion, and Conclusion."
    )
    pdf.multi_cell(0, lh, p8_text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # ===================== NOVELTY SUMMARY =====================
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, lh, "Summary of Novelty Claims (For Reviewer Response)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Times", "", 12)
    novelty_text = (
        "N1: First application of Kolmogorov-Arnold Networks (KAN) for Parkinson's disease detection "
        "from speech signals.\n"
        "N2: Novel Adaptive Tree-Spline Fusion architecture combining gradient boosting with "
        "learnable spline functions, with mathematically optimized dataset-specific weight ratios.\n"
        "N3: Dual Explainability framework - SHAP for tree-based reasoning + symbolic spline "
        "extraction from KAN for continuous feature interpretation.\n"
        "N4: Rigorous zero-leakage evaluation using Patient-wise GroupKFold, directly exposing the "
        "inflated results reported in prior works.\n"
        "N5: First multi-lingual, cross-corpus validation for PD speech detection across Turkish, "
        "Bengali, Spanish, and other datasets, proving language-independent generalization."
    )
    pdf.multi_cell(0, lh, novelty_text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # ===================== CONCLUSION =====================
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, lh, "Conclusion", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Times", "", 12)
    conc_text = (
        "By systematically executing these eight phases - eliminating data leakage, introducing "
        "Adaptive Tree-Spline Fusion, providing dual medical explainability, validating across "
        "multiple global datasets, conducting rigorous ablation and statistical significance testing, "
        "and presenting state-of-the-art comparisons - our framework will be statistically "
        "bulletproof and highly competitive for publication in a Q1 medical informatics journal."
    )
    pdf.multi_cell(0, lh, conc_text, new_x="LMARGIN", new_y="NEXT")

    pdf.output("Q1_Journal_Roadmap_Updated.pdf")

if __name__ == "__main__":
    generate_q1_roadmap_pdf()
