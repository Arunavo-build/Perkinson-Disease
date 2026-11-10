import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
matplotlib.rcParams['font.family'] = 'sans-serif'
import matplotlib.pyplot as plt
from kan import KAN
import warnings
warnings.filterwarnings('ignore')

os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def update_fig5_kan_splines():
    print("--- Updating Fig 5: KAN Symbolic B-Spline Response Curves (6 Clinically Matched Biomarkers) ---")
    df = pd.read_csv('pd_speech_features.csv', header=1)
    y = df['class'].values
    
    # Exact 6 biomarkers covering Cepstral, Energy Dynamics, Spectral Acceleration, Wavelet Turbulence, Jitter, and Shimmer
    target_features = [
        ('mean_MFCC_2nd_coef', 'mean_MFCC_2nd_coef (Formant Envelope)'),
        ('std_delta_delta_log_energy', 'std_delta_delta_log_energy (Micro-Tremor)'),
        ('std_8th_delta_delta', 'std_8th_delta_delta (Kinematic Freezing)'),
        ('tqwt_entropy_log_dec_12', 'tqwt_entropy_log_dec_12 (Glottal Turbulence)'),
        ('locPctJitter', 'locPctJitter (Pitch Perturbation / Jitter %)'),
        ('locShimmer', 'locShimmer (Amplitude Perturbation / Shimmer)')
    ]
    cols = [c[0] for c in target_features]
    titles = [c[1] for c in target_features]
    
    X_sub = df[cols].values
    scaler = StandardScaler()
    X_sub_scaled = scaler.fit_transform(X_sub)
    
    torch.manual_seed(42)
    np.random.seed(42)
    in_dim = len(cols)
    kan = KAN(width=[in_dim, 6, 2], grid=5, k=3, seed=42, auto_save=False, device=device)
    
    X_t = torch.tensor(X_sub_scaled, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.long).to(device)
    
    optimizer = optim.AdamW(kan.parameters(), lr=0.01, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(70):
        kan.train()
        optimizer.zero_grad()
        out = kan(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
        
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2), dpi=300)
    axes = axes.flatten()
    x_eval = np.linspace(-2.5, 2.5, 100)
    
    kan.eval()
    for i, (col_name, panel_title) in enumerate(target_features):
        X_probe = np.zeros((100, in_dim))
        X_probe[:, i] = x_eval
        X_probe_t = torch.tensor(X_probe, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            out_probe = kan(X_probe_t)
            prob_pd = torch.softmax(out_probe, dim=1)[:, 1].cpu().numpy()
            
        ax = axes[i]
        ax.plot(x_eval, prob_pd, color='#c62828', linewidth=2.6, label='Learned B-Spline $\\phi_i(z)$')
        ax.axhline(0.5, color='#455a64', linestyle='--', alpha=0.7, label='Decision Threshold ($P=0.5$)')
        ax.axvline(0.0, color='#90a4ae', linestyle=':', alpha=0.6)
        ax.set_title(panel_title, fontsize=10.5, fontweight='bold')
        ax.set_xlabel("Normalized Acoustic Value ($z$-score)", fontsize=10)
        ax.set_ylabel("Predicted $P(\\mathrm{PD} \\mid z)$", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        if i == 0:
            ax.legend(loc='lower right', fontsize=9)
            
    plt.suptitle("KAN Univariate Cubic B-Spline Activation Trajectories Across Core Vocal Pathologies", fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig('figures/Fig5_kan_spline_activations.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved updated figures/Fig5_kan_spline_activations.png")

def generate_fig7_domain_adaptation_and_ablation():
    print("--- Generating Fig 7: Cross-Lingual Domain Adaptation & Fusion Ablation ---")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), dpi=300)
    
    # Panel A: Cross-Lingual Domain Adaptation (Patient-Grouped 5-Fold vs Zero-Shot)
    ax1 = axes[0]
    directions = ['Turkish $\\rightarrow$ Bengali\n(27 Shared Feats)', 'Bengali $\\rightarrow$ Turkish\n(27 Shared Feats)', 'Turkish $\\rightarrow$ Oxford\n(13 Shared Feats)']
    zero_shot = [74.56, 81.88, 70.26]
    coral_only = [72.44, 80.69, 71.28]
    few_shot = [83.19, 84.59, 71.79]
    hybrid_prop = [84.67, 85.09, 76.28]
    
    x = np.arange(len(directions))
    w = 0.19
    
    b1 = ax1.bar(x - 1.5*w, zero_shot, w, label='Zero-Shot Baseline', color='#90caf9', edgecolor='#1e88e5')
    b2 = ax1.bar(x - 0.5*w, coral_only, w, label='Unsupervised CORAL', color='#ffe082', edgecolor='#ffa000')
    b3 = ax1.bar(x + 0.5*w, few_shot, w, label='Few-Shot Transfer (20%)', color='#a5d6a7', edgecolor='#43a047')
    b4 = ax1.bar(x + 1.5*w, hybrid_prop, w, label='Proposed CORAL + Few-Shot', color='#ef5350', edgecolor='#b71c1c')
    
    for bars in [b1, b2, b3, b4]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 0.6, f"{h:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
            
    ax1.set_xticks(x)
    ax1.set_xticklabels(directions, fontsize=10, fontweight='bold')
    ax1.set_ylim(60, 92)
    ax1.set_ylabel('Cross-Corpus Test Accuracy (%)', fontsize=11)
    ax1.set_title('(a) Cross-Lingual Domain Adaptation Progression', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # Panel B: Feature Fusion vs. Model Fusion & 10-Model Benchmark on Turkish UCI
    ax2 = axes[1]
    methods = [
        'TabNet\n(Deep Tabular)',
        'TabTransformer\n(Attention)',
        'Baseline Feats\n(Perturbation)',
        'MFCC Feats\n(Cepstral)',
        'SAINT\n(Row+Col Attn)',
        'TQWT Feats\n(Wavelet)',
        'KAN\n(Standalone)',
        'Feature Fusion\n(Concat 1+2+3)',
        'LightGBM\n(Standalone)',
        'Adaptive Fusion\n(Proposed)'
    ]
    accs = [71.96, 75.91, 76.06, 77.77, 80.41, 81.22, 81.86, 82.42, 84.26, 85.18]
    colors = ['#cfd8dc', '#cfd8dc', '#b3e5fc', '#b3e5fc', '#cfd8dc', '#b3e5fc', '#ffcc80', '#81d4fa', '#a5d6a7', '#e53935']
    
    y_pos = np.arange(len(methods))
    bars = ax2.barh(y_pos, accs, color=colors, edgecolor='#37474f', height=0.68)
    for bar, v in zip(bars, accs):
        ax2.text(v + 0.3, bar.get_y() + bar.get_height()/2, f"{v:.2f}%", va='center', ha='left', fontsize=9, fontweight='bold')
        
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(methods, fontsize=9.5)
    ax2.set_xlim(65, 89)
    ax2.set_xlabel('Patient-Wise 5-Fold GroupKFold Accuracy (%)', fontsize=11)
    ax2.set_title('(b) Architectural & Modality Benchmark (Turkish UCI)', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.suptitle("Cross-Lingual Domain Alignment and Multi-Paradigm Architectural Ablation", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figures/Fig7_Domain_Adaptation_and_Ablation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved figures/Fig7_Domain_Adaptation_and_Ablation.png")

if __name__ == '__main__':
    update_fig5_kan_splines()
    generate_fig7_domain_adaptation_and_ablation()
