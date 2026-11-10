import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import lightgbm as lgb
from kan import KAN
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_kan(X_tr, y_tr, X_te, epochs=80, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    in_dim = X_tr.shape[1]
    kan = KAN(width=[in_dim, 8, 2], grid=5, k=3, seed=seed, device=device)
    
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long).to(device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32).to(device)
    
    optimizer = optim.Adam(kan.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        kan.train()
        optimizer.zero_grad()
        out = kan(X_tr_t)
        loss = criterion(out, y_tr_t)
        loss.backward()
        optimizer.step()
        
    kan.eval()
    with torch.no_grad():
        logits_te = kan(X_te_t)
        probs_te = torch.softmax(logits_te, dim=1).cpu().numpy()
    return probs_te

def evaluate_pair(df_train, y_train, df_test, y_test, matched_cols, train_name, test_name, alpha=0.6):
    cols_train = [p[0] for p in matched_cols]
    cols_test = [p[1] for p in matched_cols]
    
    X_train = df_train[cols_train].values
    X_test = df_test[cols_test].values
    
    # Domain-specific standardization to eliminate microphone gain and recording environment offset
    scaler_tr = StandardScaler()
    X_train_s = scaler_tr.fit_transform(X_train)
    scaler_te = StandardScaler()
    X_test_s = scaler_te.fit_transform(X_test)
    
    # 1. LightGBM
    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    clf.fit(X_train_s, y_train)
    p_lgb = clf.predict_proba(X_test_s)
    
    # 2. KAN
    p_kan = train_kan(X_train_s, y_train, X_test_s, epochs=70)
    
    # 3. Adaptive Tree-Spline Fusion
    p_fuse = alpha * p_lgb + (1.0 - alpha) * p_kan
    y_pred_fuse = np.argmax(p_fuse, axis=1)
    
    acc_lgb = accuracy_score(y_test, np.argmax(p_lgb, axis=1))
    acc_kan = accuracy_score(y_test, np.argmax(p_kan, axis=1))
    acc_fuse = accuracy_score(y_test, y_pred_fuse)
    f1_fuse = f1_score(y_test, y_pred_fuse, zero_division=0)
    rec_fuse = recall_score(y_test, y_pred_fuse, zero_division=0)
    try:
        auc_fuse = roc_auc_score(y_test, p_fuse[:, 1])
    except:
        auc_fuse = 0.5
        
    print(f"\n[{train_name} -> {test_name}] (Common Features: {len(matched_cols)}):")
    print(f"  LightGBM Acc: {acc_lgb*100:.2f}% | KAN Acc: {acc_kan*100:.2f}% | Adaptive Fusion Acc: {acc_fuse*100:.2f}% | F1: {f1_fuse*100:.2f}% | AUC: {auc_fuse:.4f}")
    
    return {
        'source_dataset': train_name,
        'target_dataset': test_name,
        'n_common_features': len(matched_cols),
        'accuracy_lgb': acc_lgb,
        'accuracy_kan': acc_kan,
        'accuracy_fusion': acc_fuse,
        'f1_fusion': f1_fuse,
        'sensitivity_fusion': rec_fuse,
        'auc_fusion': auc_fuse
    }

def main():
    print("=== STARTING PHASE 5: MULTI-DATASET CROSS-LINGUAL GENERALIZATION ===")
    
    # Load 3 datasets
    df_tr = pd.read_csv('pd_speech_features.csv', header=1)
    y_tr = df_tr['class'].values
    
    df_bn = pd.read_csv('BenSParX/BenSParX.csv')
    y_bn = df_bn['class'].values
    
    df_ox = pd.read_csv('oxford_parkinsons.csv')
    y_ox = df_ox['status'].values
    
    results = []
    
    # 1. Turkish <-> Bengali (27 identical acoustic features)
    common_tr_bn = [c for c in df_tr.columns if c in df_bn.columns and c not in ['id', 'class']]
    matched_tr_bn = [(c, c) for c in common_tr_bn]
    matched_bn_tr = [(c, c) for c in common_tr_bn]
    results.append(evaluate_pair(df_tr, y_tr, df_bn, y_bn, matched_tr_bn, 'Turkish (UCI)', 'Bengali (BenSParX)', alpha=0.6))
    results.append(evaluate_pair(df_bn, y_bn, df_tr, y_tr, matched_bn_tr, 'Bengali (BenSParX)', 'Turkish (UCI)', alpha=0.8))
    
    # 2. Turkish <-> Oxford (Standard acoustic domain mapping: Jitter, Shimmer, RPDE, DFA, PPE)
    ox_to_tr = [
        ('MDVP:Jitter(%)', 'locPctJitter'),
        ('MDVP:Jitter(Abs)', 'locAbsJitter'),
        ('MDVP:RAP', 'rapJitter'),
        ('MDVP:PPQ', 'ppq5Jitter'),
        ('Jitter:DDP', 'ddpJitter'),
        ('MDVP:Shimmer', 'locShimmer'),
        ('MDVP:Shimmer(dB)', 'locDbShimmer'),
        ('Shimmer:APQ3', 'apq3Shimmer'),
        ('Shimmer:APQ5', 'apq5Shimmer'),
        ('Shimmer:DDA', 'ddaShimmer'),
        ('RPDE', 'RPDE'),
        ('DFA', 'DFA'),
        ('PPE', 'PPE')
    ]
    matched_tr_ox = [(tr, ox) for ox, tr in ox_to_tr if tr in df_tr.columns and ox in df_ox.columns]
    matched_ox_tr = [(ox, tr) for ox, tr in ox_to_tr if tr in df_tr.columns and ox in df_ox.columns]
    results.append(evaluate_pair(df_tr, y_tr, df_ox, y_ox, matched_tr_ox, 'Turkish (UCI)', 'Oxford (English)', alpha=0.6))
    results.append(evaluate_pair(df_ox, y_ox, df_tr, y_tr, matched_ox_tr, 'Oxford (English)', 'Turkish (UCI)', alpha=0.7))
    
    # 3. Bengali <-> Oxford (Clean 10 perturbation features: 5 Jitter + 5 Shimmer)
    ox_to_bn = [
        ('MDVP:Jitter(%)', 'locPctJitter'),
        ('MDVP:Jitter(Abs)', 'locAbsJitter'),
        ('MDVP:RAP', 'rapJitter'),
        ('MDVP:PPQ', 'ppq5Jitter'),
        ('Jitter:DDP', 'ddpJitter'),
        ('MDVP:Shimmer', 'locShimmer'),
        ('MDVP:Shimmer(dB)', 'locDbShimmer'),
        ('Shimmer:APQ3', 'apq3Shimmer'),
        ('Shimmer:APQ5', 'apq5Shimmer'),
        ('Shimmer:DDA', 'ddaShimmer')
    ]
    matched_bn_ox = [(bn, ox) for ox, bn in ox_to_bn if bn in df_bn.columns and ox in df_ox.columns]
    matched_ox_bn = [(ox, bn) for ox, bn in ox_to_bn if bn in df_bn.columns and ox in df_ox.columns]
    results.append(evaluate_pair(df_bn, y_bn, df_ox, y_ox, matched_bn_ox, 'Bengali (BenSParX)', 'Oxford (English)', alpha=0.6))
    results.append(evaluate_pair(df_ox, y_ox, df_bn, y_bn, matched_ox_bn, 'Oxford (English)', 'Bengali (BenSParX)', alpha=0.7))
    
    res_df = pd.DataFrame(results)
    res_df.to_csv('results/phase5_cross_dataset_summary.csv', index=False)
    print("\nPhase 5 Cross-Dataset Generalization completed and saved to results/phase5_cross_dataset_summary.csv!")

if __name__ == '__main__':
    main()
