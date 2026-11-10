import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
matplotlib.rcParams['font.family'] = 'sans-serif'
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

from matplotlib.patches import FancyBboxPatch

def generate_system_architecture():
    print("\n--- 1. Generating Publication System Architecture Diagram ---")
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    ax.axis('off')
    
    # Palette
    c_input = '#e3f2fd'
    c_tree = '#e8f5e9'
    c_kan = '#fff3e0'
    c_fuse = '#f3e5f5'
    c_out = '#ffebee'
    
    # Boxes
    boxes = [
        # (x, y, w, h, text, color, title)
        (0.03, 0.35, 0.16, 0.35, "Raw Speech Audio\n& Multi-Corpus\n- Turkish UCI (754 feats)\n- Bengali (71 feats)\n- Oxford UCI (22 feats)", c_input, "Multi-Lingual Acoustic Inputs"),
        (0.24, 0.60, 0.22, 0.30, "Gradient Boosted Trees\n(LightGBM)\n- Discrete axis-aligned splits\n- Robust against outliers\n- High tabular capacity", c_tree, "Path A: Tree Ensembling"),
        (0.24, 0.15, 0.22, 0.30, "Kolmogorov-Arnold Network\n(KAN)\n- Univariate B-Spline activations\n- Smooth continuous dynamics\n- Symbolic interpretability", c_kan, "Path B: Spline Deep Learning"),
        (0.53, 0.35, 0.22, 0.35, "Adaptive Weighting Engine\n$P_{fuse} = \\alpha P_{tree} + (1-\\alpha) P_{spline}$\n- Dynamically adapted via inner CV\n- High-dim: $\\alpha \\approx 0.60$\n- Low-dim: $\\alpha \\approx 0.50$", c_fuse, "Adaptive Tree-Spline Fusion"),
        (0.81, 0.35, 0.16, 0.35, "Clinical Diagnosis\n- Zero-Leakage GroupKFold\n- Dual XAI (SHAP + Splines)\n- Cross-Dataset Generalization", c_out, "Screening Output & XAI")
    ]
    
    for x, y, w, h, text, color, title in boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=color, ec='#37474f', lw=1.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.05, title, ha='center', va='top', fontsize=10.5, fontweight='bold', color='#263238')
        ax.text(x + w/2, y + h/2 - 0.03, text, ha='center', va='center', fontsize=9, color='#37474f')
        
    # Connective arrows
    arrow_props = dict(facecolor='#37474f', edgecolor='#37474f', width=1.5, headwidth=7)
    
    # Input -> Tree & Spline
    ax.annotate('', xy=(0.24, 0.75), xytext=(0.19, 0.55), arrowprops=arrow_props)
    ax.annotate('', xy=(0.24, 0.30), xytext=(0.19, 0.50), arrowprops=arrow_props)
    
    # Tree & Spline -> Fusion
    ax.annotate('', xy=(0.53, 0.60), xytext=(0.46, 0.72), arrowprops=arrow_props)
    ax.annotate('', xy=(0.53, 0.45), xytext=(0.46, 0.32), arrowprops=arrow_props)
    
    # Fusion -> Output
    ax.annotate('', xy=(0.81, 0.525), xytext=(0.75, 0.525), arrowprops=arrow_props)
    
    plt.title("Proposed Adaptive Tree-Spline Hybrid Framework for Parkinson's Disease Voice Screening", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('figures/Fig1_System_Architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved figures/Fig1_System_Architecture.png")

def generate_roc_curves():
    print("\n--- 2. Generating Multi-Dataset ROC Comparison Curves ---")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    
    datasets = [
        ('turkish_uci', 'Turkish UCI Dataset', axes[0]),
        ('bengali_bensparx', 'Bengali BenSParX Dataset', axes[1]),
        ('oxford_uci', 'Oxford UCI Dataset', axes[2])
    ]
    
    colors = {'LightGBM': '#1f77b4', 'KAN': '#ff7f0e', 'Naive_Fusion': '#2ca02c', 'Adaptive_Fusion': '#d62728'}
    
    for ds_key, ds_title, ax in datasets:
        pred_file = f"results/predictions_{ds_key}.npz"
        if not os.path.exists(pred_file):
            print(f"Skipping {ds_key}: file not found.")
            continue
            
        data = np.load(pred_file)
        y_true = data['y_true']
        
        models_dict = {
            'LightGBM': data['oof_lgb'],
            'KAN': data['oof_kan'],
            'Naive_Fusion': data['oof_naive'],
            'Adaptive_Fusion (Proposed)': data['oof_adapt']
        }
        
        for m_name, probs in models_dict.items():
            fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
            roc_auc = auc(fpr, tpr)
            c = colors.get(m_name.split()[0], '#d62728')
            lw = 2.5 if 'Adaptive' in m_name else 1.8
            ls = '-' if 'Adaptive' in m_name else '--'
            ax.plot(fpr, tpr, color=c, lw=lw, linestyle=ls, label=f"{m_name} (AUC = {roc_auc:.3f})")
            
        ax.plot([0, 1], [0, 1], color='navy', lw=1.2, linestyle=':')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)
        ax.set_title(ds_title, fontsize=12, fontweight='bold')
        ax.legend(loc="lower right", fontsize=9.5)
        ax.grid(True, alpha=0.3)
        
    plt.suptitle("Receiver Operating Characteristic (ROC) Curves Across Multiple Linguistic Corpora", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figures/Fig2_MultiDataset_ROC_Curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved figures/Fig2_MultiDataset_ROC_Curves.png")

def generate_confusion_matrices():
    print("\n--- 3. Generating Normalized Confusion Matrices ---")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=300)
    
    datasets = [
        ('turkish_uci', 'Turkish UCI', axes[0]),
        ('bengali_bensparx', 'Bengali BenSParX', axes[1]),
        ('oxford_uci', 'Oxford UCI', axes[2])
    ]
    
    for ds_key, ds_title, ax in datasets:
        pred_file = f"results/predictions_{ds_key}.npz"
        if not os.path.exists(pred_file):
            continue
            
        data = np.load(pred_file)
        y_true = data['y_true']
        y_pred = np.argmax(data['oof_adapt'], axis=1)
        
        cm = confusion_matrix(y_true, y_pred, normalize='true') * 100
        cm_counts = confusion_matrix(y_true, y_pred)
        
        labels = [f"{cm[i, j]:.1f}%\n({cm_counts[i, j]})" for i in range(2) for j in range(2)]
        labels = np.asarray(labels).reshape(2, 2)
        
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False, ax=ax, annot_kws={"size": 11, "weight": "bold"})
        ax.set_title(f"{ds_title}\n(Proposed Adaptive Fusion)", fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)
        ax.set_xticklabels(['Healthy Control', 'Parkinson\'s'])
        ax.set_yticklabels(['Healthy Control', 'Parkinson\'s'])
        
    plt.suptitle("Normalized Confusion Matrices under Patient-wise GroupKFold", fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig('figures/Fig3_Confusion_Matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved figures/Fig3_Confusion_Matrices.png")

def generate_cross_dataset_transfer_heatmap():
    print("\n--- 4. Generating Cross-Dataset Transfer Heatmap ---")
    csv_file = 'results/phase5_cross_dataset_summary.csv'
    if not os.path.exists(csv_file):
        print("Skipping cross-dataset heatmap: CSV not found.")
        return
        
    df = pd.read_csv(csv_file)
    
    # Create transfer matrix
    datasets = ['Turkish (UCI)', 'Bengali (BenSParX)', 'Oxford (English)']
    matrix = pd.DataFrame(index=datasets, columns=datasets, dtype=float)
    
    # Fill self with within-dataset benchmark accuracies from phase 3
    bench_file = 'results/phase3_benchmark_summary.csv'
    if os.path.exists(bench_file):
        bdf = pd.read_csv(bench_file)
        # Adapt names
        name_map = {'Turkish_UCI': 'Turkish (UCI)', 'Bengali_BenSParX': 'Bengali (BenSParX)', 'Oxford_UCI': 'Oxford (English)'}
        for _, r in bdf[bdf['model'] == 'Adaptive_Fusion'].iterrows():
            matrix.loc[name_map[r['dataset']], name_map[r['dataset']]] = r['accuracy_mean'] * 100
            
    for _, r in df.iterrows():
        src = r['source_dataset']
        tgt = r['target_dataset']
        if src in datasets and tgt in datasets:
            matrix.loc[src, tgt] = r['accuracy_fusion'] * 100
            
    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(matrix, annot=True, fmt='.1f', cmap='YlGnBu', cbar_kws={'label': 'Test Accuracy (%)'}, vmin=50, vmax=100)
    plt.title("Cross-Dataset & Cross-Lingual Generalization Matrix (%)\n(Train on Row Dataset $\\rightarrow$ Test on Column Dataset)", fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Target Evaluation Dataset', fontsize=11)
    plt.ylabel('Source Training Dataset', fontsize=11)
    plt.tight_layout()
    plt.savefig('figures/Fig6_Cross_Dataset_Heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved figures/Fig6_Cross_Dataset_Heatmap.png")

def generate_latex_and_markdown_tables():
    print("\n--- 5. Generating LaTeX & Markdown Manuscript Tables ---")
    bench_file = 'results/phase3_benchmark_summary.csv'
    if not os.path.exists(bench_file):
        return
        
    bdf = pd.read_csv(bench_file)
    
    # Generate Markdown Table
    md_lines = [
        "# Comprehensive Benchmark Results Across Three Multilingual Corpora",
        "",
        "| Dataset | Model | Accuracy (Mean ± Std) | Sensitivity (Recall) | Specificity | F1-Score | ROC-AUC |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |"
    ]
    
    for _, r in bdf.iterrows():
        ds = r['dataset'].replace('_', ' ')
        m = r['model'].replace('_', ' ')
        acc = f"{r['accuracy_mean']*100:.2f} ± {r['accuracy_std']*100:.2f}%"
        sens = f"{r['sensitivity_mean']*100:.2f}%"
        spec = f"{r['specificity_mean']*100:.2f}%"
        f1 = f"{r['f1_mean']*100:.2f}%"
        auc_val = f"{r['auc_mean']:.4f}"
        
        # Bold proposed model
        if 'Adaptive' in m:
            m = f"**{m} (Ours)**"
            acc = f"**{acc}**"
            sens = f"**{sens}**"
            spec = f"**{spec}**"
            f1 = f"**{f1}**"
            auc_val = f"**{auc_val}**"
            
        md_lines.append(f"| {ds} | {m} | {acc} | {sens} | {spec} | {f1} | {auc_val} |")
        
    with open('results/Table1_Benchmark_Summary.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    print("Saved results/Table1_Benchmark_Summary.md")

def main():
    print("=== STARTING PHASE 8: PUBLICATION FIGURES & MANUSCRIPT TABLES ===")
    generate_system_architecture()
    generate_roc_curves()
    generate_confusion_matrices()
    generate_cross_dataset_transfer_heatmap()
    generate_latex_and_markdown_tables()
    print("\nPhase 8 Publication Artifacts generated successfully!")

if __name__ == '__main__':
    main()
