import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, confusion_matrix
)
import lightgbm as lgb
from kan import KAN
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_kan_model(X_train, y_train, X_test, seed=42, epochs=60):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    in_dim = X_train.shape[1]
    kan = KAN(width=[in_dim, 8, 2], grid=5, k=3, seed=seed, device=device)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    optimizer = optim.Adam(kan.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        kan.train()
        optimizer.zero_grad()
        out = kan(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()
        
    kan.eval()
    with torch.no_grad():
        logits_test = kan(X_test_t)
        probs_test = torch.softmax(logits_test, dim=1).cpu().numpy()
        
        # also get train predictions for inner validation if needed
        logits_train = kan(X_train_t)
        probs_train = torch.softmax(logits_train, dim=1).cpu().numpy()
        
    return kan, probs_test, probs_train

def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0) # Sensitivity
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob[:, 1])
    except:
        auc = 0.5
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        'accuracy': acc,
        'sensitivity': rec,
        'specificity': spec,
        'precision': prec,
        'f1': f1,
        'auc': auc
    }

def run_dataset_benchmark(X, y, groups, dataset_name):
    print(f"\n==================================================================")
    print(f"Phase 2/3: Patient-wise 5-Fold GroupKFold Benchmark: {dataset_name}")
    print(f"Total Samples: {len(y)}, Features: {X.shape[1]}, Unique Subjects: {len(np.unique(groups))}")
    print(f"==================================================================")
    
    gkf = GroupKFold(n_splits=5)
    
    models = ['LightGBM', 'KAN', 'Naive_Fusion', 'Adaptive_Fusion']
    fold_metrics = {m: [] for m in models}
    
    oof_probs_lgb = np.zeros((len(y), 2))
    oof_probs_kan = np.zeros((len(y), 2))
    oof_probs_naive = np.zeros((len(y), 2))
    oof_probs_adapt = np.zeros((len(y), 2))
    oof_chosen_alphas = []
    
    alpha_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        print(f"\n--- Fold {fold + 1} / 5 ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]
        
        # Scaling strictly fitted on train
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # 1. Train LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42 + fold,
            verbose=-1
        )
        lgb_model.fit(X_train_s, y_train)
        probs_lgb = lgb_model.predict_proba(X_test_s)
        probs_train_lgb = lgb_model.predict_proba(X_train_s)
        
        # 2. Train KAN
        _, probs_kan, probs_train_kan = train_kan_model(
            X_train_s, y_train, X_test_s, seed=42 + fold, epochs=60
        )
        
        # 3. Naive Soft Voting (50-50)
        probs_naive = 0.5 * probs_lgb + 0.5 * probs_kan
        
        # 4. Adaptive Tree-Spline Weighted Fusion (Inner-Fold Weight Tuning)
        # Using inner 3-fold GroupKFold to find optimal alpha without test leakage
        inner_gkf = GroupKFold(n_splits=3)
        inner_scores = {a: [] for a in alpha_candidates}
        
        for in_train, in_val in inner_gkf.split(X_train_s, y_train, groups=groups_train):
            in_X_tr, in_y_tr = X_train_s[in_train], y_train[in_train]
            in_X_va, in_y_va = X_train_s[in_val], y_train[in_val]
            
            # Quick inner LGBM
            m_lgb = lgb.LGBMClassifier(n_estimators=60, learning_rate=0.05, random_state=42, verbose=-1)
            m_lgb.fit(in_X_tr, in_y_tr)
            p_lgb_val = m_lgb.predict_proba(in_X_va)
            
            # Quick inner KAN
            _, p_kan_val, _ = train_kan_model(in_X_tr, in_y_tr, in_X_va, seed=100, epochs=40)
            
            for a in alpha_candidates:
                p_fuse = a * p_lgb_val + (1.0 - a) * p_kan_val
                pred_fuse = np.argmax(p_fuse, axis=1)
                inner_scores[a].append(accuracy_score(in_y_va, pred_fuse))
                
        best_alpha = max(alpha_candidates, key=lambda a: np.mean(inner_scores[a]))
        oof_chosen_alphas.append(best_alpha)
        
        probs_adapt = best_alpha * probs_lgb + (1.0 - best_alpha) * probs_kan
        
        # Save OOF
        oof_probs_lgb[test_idx] = probs_lgb
        oof_probs_kan[test_idx] = probs_kan
        oof_probs_naive[test_idx] = probs_naive
        oof_probs_adapt[test_idx] = probs_adapt
        
        # Compute metrics per fold
        fold_metrics['LightGBM'].append(compute_metrics(y_test, np.argmax(probs_lgb, axis=1), probs_lgb))
        fold_metrics['KAN'].append(compute_metrics(y_test, np.argmax(probs_kan, axis=1), probs_kan))
        fold_metrics['Naive_Fusion'].append(compute_metrics(y_test, np.argmax(probs_naive, axis=1), probs_naive))
        fold_metrics['Adaptive_Fusion'].append(compute_metrics(y_test, np.argmax(probs_adapt, axis=1), probs_adapt))
        
        print(f"  Inner-selected optimal alpha: {best_alpha:.2f} (LGBM: {best_alpha*100:.0f}%, KAN: {(1-best_alpha)*100:.0f}%)")
        print(f"  Fold Accuracies -> LGBM: {fold_metrics['LightGBM'][-1]['accuracy']*100:.2f}%, KAN: {fold_metrics['KAN'][-1]['accuracy']*100:.2f}%, Adaptive: {fold_metrics['Adaptive_Fusion'][-1]['accuracy']*100:.2f}%")

    # Overall Summary
    print(f"\n=======================================================")
    print(f"Summary Results for {dataset_name} (Patient-wise 5-Fold)")
    print(f"=======================================================")
    
    summary_rows = []
    for m in models:
        accs = [fm['accuracy'] for fm in fold_metrics[m]]
        recs = [fm['sensitivity'] for fm in fold_metrics[m]]
        specs = [fm['specificity'] for fm in fold_metrics[m]]
        f1s = [fm['f1'] for fm in fold_metrics[m]]
        aucs = [fm['auc'] for fm in fold_metrics[m]]
        
        print(f"{m:<18}: Acc={np.mean(accs)*100:.2f}±{np.std(accs)*100:.2f}% | Sens={np.mean(recs)*100:.2f}% | Spec={np.mean(specs)*100:.2f}% | F1={np.mean(f1s)*100:.2f}% | AUC={np.mean(aucs):.4f}")
        
        summary_rows.append({
            'dataset': dataset_name,
            'model': m,
            'accuracy_mean': np.mean(accs),
            'accuracy_std': np.std(accs),
            'sensitivity_mean': np.mean(recs),
            'specificity_mean': np.mean(specs),
            'f1_mean': np.mean(f1s),
            'auc_mean': np.mean(aucs),
            'selected_alphas': str(oof_chosen_alphas)
        })
        
    # Save predictions file for downstream ROC, confusion matrices, McNemar
    np.savez_compressed(
        f"results/predictions_{dataset_name.lower().replace(' ', '_')}.npz",
        y_true=y,
        groups=groups,
        oof_lgb=oof_probs_lgb,
        oof_kan=oof_probs_kan,
        oof_naive=oof_probs_naive,
        oof_adapt=oof_probs_adapt,
        alphas=np.array(oof_chosen_alphas)
    )
    
    return summary_rows

def main():
    all_summaries = []
    
    # 1. Turkish UCI Dataset
    df_tr = pd.read_csv('pd_speech_features.csv', header=1)
    y_tr = df_tr['class'].values
    groups_tr = df_tr['id'].values
    X_tr = df_tr.drop(columns=['class', 'id']).values
    all_summaries.extend(run_dataset_benchmark(X_tr, y_tr, groups_tr, 'Turkish_UCI'))
    
    # 2. Bengali BenSParX Dataset
    df_bn = pd.read_csv('BenSParX/BenSParX.csv')
    y_bn = df_bn['class'].values
    groups_bn = df_bn['id'].astype(str).apply(lambda x: x.split('_')[0]).values
    X_bn = df_bn.drop(columns=['class', 'id']).values
    all_summaries.extend(run_dataset_benchmark(X_bn, y_bn, groups_bn, 'Bengali_BenSParX'))
    
    # 3. Oxford UCI Dataset
    df_ox = pd.read_csv('oxford_parkinsons.csv')
    y_ox = df_ox['status'].values
    groups_ox = df_ox['name'].apply(lambda x: '_'.join(x.split('_')[:3])).values
    X_ox = df_ox.drop(columns=['name', 'status']).values
    all_summaries.extend(run_dataset_benchmark(X_ox, y_ox, groups_ox, 'Oxford_UCI'))
    
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv('results/phase3_benchmark_summary.csv', index=False)
    print("\nPhase 2 & 3 Benchmarking completed and saved to results/phase3_benchmark_summary.csv!")

if __name__ == '__main__':
    main()
