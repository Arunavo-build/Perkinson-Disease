import os
import numpy as np
import pandas as pd
import scipy.linalg
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)

# ---------------------------------------------------------
# CORAL (Correlation Alignment) Algorithm for Domain Adaptation
# ---------------------------------------------------------
def coral_fit_transform(Xs, Xt):
    """
    Aligns the covariance of Source (Xs) to match the Target (Xt).
    """
    cov_s = np.cov(Xs, rowvar=False) + np.eye(Xs.shape[1]) * 1e-5
    cov_t = np.cov(Xt, rowvar=False) + np.eye(Xt.shape[1]) * 1e-5
    
    # Whitening Source
    A_s = scipy.linalg.inv(scipy.linalg.sqrtm(cov_s))
    Xs_whitened = Xs.dot(A_s)
    
    # Coloring with Target covariance
    A_t = scipy.linalg.sqrtm(cov_t)
    Xs_coral = Xs_whitened.dot(A_t)
    
    return np.real(Xs_coral)

def evaluate_novel_approaches(df_source, y_source, df_target, y_target, matched_cols, source_name, target_name):
    print(f"\n=======================================================")
    print(f"Cross-Lingual Direction: {source_name} -> {target_name}")
    print(f"Features: {len(matched_cols)} common acoustic features")
    print(f"=======================================================")
    
    cols_s = [p[0] for p in matched_cols]
    cols_t = [p[1] for p in matched_cols]
    
    Xs = df_source[cols_s].values
    Xt = df_target[cols_t].values
    
    # Standardize both domains
    scaler_s = StandardScaler()
    Xs_s = scaler_s.fit_transform(Xs)
    
    scaler_t = StandardScaler()
    Xt_s = scaler_t.fit_transform(Xt)
    
    # ---------------------------------------------------------
    # Approach 1: Baseline Zero-Shot
    # ---------------------------------------------------------
    clf_base = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    clf_base.fit(Xs_s, y_source)
    preds_base = clf_base.predict(Xt_s)
    acc_base = accuracy_score(y_target, preds_base)
    print(f"1. Baseline (Zero-Shot) Accuracy:      {acc_base*100:.2f}%")
    
    # ---------------------------------------------------------
    # Approach 2: Unsupervised Domain Adaptation (CORAL)
    # ---------------------------------------------------------
    Xs_coral = coral_fit_transform(Xs_s, Xt_s)
    clf_coral = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    clf_coral.fit(Xs_coral, y_source)
    preds_coral = clf_coral.predict(Xt_s)
    acc_coral = accuracy_score(y_target, preds_coral)
    print(f"2. Domain Adaptation (CORAL) Accuracy: {acc_coral*100:.2f}%")
    
    # ---------------------------------------------------------
    # Approach 3: Few-Shot Transfer Learning (20% Target data)
    # ---------------------------------------------------------
    Xt_train, Xt_test, yt_train, yt_test = train_test_split(Xt_s, y_target, test_size=0.8, random_state=42, stratify=y_target)
    
    X_transfer = np.vstack((Xs_s, Xt_train))
    y_transfer = np.concatenate((y_source, yt_train))
    
    clf_transfer = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    clf_transfer.fit(X_transfer, y_transfer)
    preds_transfer = clf_transfer.predict(Xt_test)
    acc_transfer = accuracy_score(yt_test, preds_transfer)
    print(f"3. Few-Shot Transfer (20% Target):     {acc_transfer*100:.2f}%")
    
    # ---------------------------------------------------------
    # Approach 4: The Q1 Proposed Novelty (CORAL + Few-Shot)
    # ---------------------------------------------------------
    Xs_coral_fs = coral_fit_transform(Xs_s, Xt_train)
    X_novel = np.vstack((Xs_coral_fs, Xt_train))
    y_novel = np.concatenate((y_source, yt_train))
    
    clf_novel = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    clf_novel.fit(X_novel, y_novel)
    preds_novel = clf_novel.predict(Xt_test)
    acc_novel = accuracy_score(yt_test, preds_novel)
    print(f"4. Proposed (CORAL + Few-Shot):        {acc_novel*100:.2f}%")
    
    return {
        'Source': source_name,
        'Target': target_name,
        'Baseline': acc_base,
        'CORAL': acc_coral,
        'Transfer': acc_transfer,
        'Proposed': acc_novel
    }

def main():
    print("Loading datasets...")
    df_tr = pd.read_csv('pd_speech_features.csv', header=1)
    y_tr = df_tr['class'].values
    
    df_bn = pd.read_csv('BenSParX/BenSParX.csv')
    y_bn = df_bn['class'].values
    
    df_ox = pd.read_csv('oxford_parkinsons.csv')
    y_ox = df_ox['status'].values
    
    results = []
    
    # Experiment A: Turkish -> Bengali
    common_tr_bn = [c for c in df_tr.columns if c in df_bn.columns and c not in ['id', 'class']]
    matched_tr_bn = [(c, c) for c in common_tr_bn]
    results.append(evaluate_novel_approaches(df_tr, y_tr, df_bn, y_bn, matched_tr_bn, 'Turkish (UCI)', 'Bengali (BenSParX)'))
    
    # Experiment B: Turkish -> Oxford
    ox_to_tr = [
        ('MDVP:Jitter(%)', 'locPctJitter'), ('MDVP:Jitter(Abs)', 'locAbsJitter'),
        ('MDVP:RAP', 'rapJitter'), ('MDVP:PPQ', 'ppq5Jitter'), ('Jitter:DDP', 'ddpJitter'),
        ('MDVP:Shimmer', 'locShimmer'), ('MDVP:Shimmer(dB)', 'locDbShimmer'),
        ('Shimmer:APQ3', 'apq3Shimmer'), ('Shimmer:APQ5', 'apq5Shimmer'),
        ('Shimmer:DDA', 'ddaShimmer'), ('RPDE', 'RPDE'), ('DFA', 'DFA'), ('PPE', 'PPE')
    ]
    matched_tr_ox = [(tr, ox) for ox, tr in ox_to_tr if tr in df_tr.columns and ox in df_ox.columns]
    results.append(evaluate_novel_approaches(df_tr, y_tr, df_ox, y_ox, matched_tr_ox, 'Turkish (UCI)', 'Oxford (English)'))
    
    # Save results
    res_df = pd.DataFrame(results)
    res_df.to_csv('results/phase9_q1_novelty_benchmark.csv', index=False)
    print("\nBenchmark saved to results/phase9_q1_novelty_benchmark.csv")

if __name__ == '__main__':
    main()
