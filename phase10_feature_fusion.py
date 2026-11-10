import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Loading UCI Turkish Parkinson's Dataset for Feature Fusion Experiment...")
    df = pd.read_csv('pd_speech_features.csv', header=1)
    
    y = df['class'].values
    groups = df['id'].values
    
    all_cols = df.columns.tolist()
    
    # Define Feature Sets
    # 1. Baseline Perturbation Features (Jitter, Shimmer, etc.)
    baseline_keywords = ['Jitter', 'Shimmer', 'PPE', 'DFA', 'RPDE', 'Pulses']
    cols_baseline = [c for c in all_cols if any(k in c for k in baseline_keywords)]
    
    # 2. MFCC Features
    cols_mfcc = [c for c in all_cols if 'MFCC' in c]
    
    # 3. TQWT (Wavelet) Features
    cols_tqwt = [c for c in all_cols if 'tqwt' in c]
    
    print(f"Feature Set 1 (Baseline): {len(cols_baseline)} features")
    print(f"Feature Set 2 (MFCC):     {len(cols_mfcc)} features")
    print(f"Feature Set 3 (TQWT):     {len(cols_tqwt)} features")
    
    X_base = df[cols_baseline].values
    X_mfcc = df[cols_mfcc].values
    X_tqwt = df[cols_tqwt].values
    
    # Feature Fusion (Concatenation)
    X_fused = np.hstack((X_base, X_mfcc, X_tqwt))
    print(f"Feature Fusion (All 3):   {X_fused.shape[1]} features")
    
    gkf = GroupKFold(n_splits=5)
    
    acc_base, acc_mfcc, acc_tqwt, acc_fused = [], [], [], []
    
    for train_idx, test_idx in gkf.split(X_fused, y, groups=groups):
        y_train, y_test = y[train_idx], y[test_idx]
        
        def evaluate_set(X):
            X_train, X_test = X[train_idx], X[test_idx]
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
            clf.fit(X_train_s, y_train)
            preds = clf.predict(X_test_s)
            return accuracy_score(y_test, preds)
            
        acc_base.append(evaluate_set(X_base))
        acc_mfcc.append(evaluate_set(X_mfcc))
        acc_tqwt.append(evaluate_set(X_tqwt))
        acc_fused.append(evaluate_set(X_fused))
        
    print("\n=======================================================")
    print("Feature Fusion Results (5-Fold Patient-Wise CV)")
    print("=======================================================")
    print(f"1. Baseline Features Only: {np.mean(acc_base)*100:.2f}%")
    print(f"2. MFCC Features Only:     {np.mean(acc_mfcc)*100:.2f}%")
    print(f"3. TQWT Features Only:     {np.mean(acc_tqwt)*100:.2f}%")
    print(f"-------------------------------------------------------")
    print(f"4. FEATURE FUSION (1+2+3): {np.mean(acc_fused)*100:.2f}%")
    print("=======================================================")

if __name__ == '__main__':
    main()
