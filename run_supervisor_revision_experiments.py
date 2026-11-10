import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest, chi2
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, matthews_corrcoef, brier_score_loss, roc_curve
)
import lightgbm as lgb
from kan import KAN
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

os.makedirs('results', exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEEDS = [42, 100, 2024, 777, 999]

def train_kan_probs(X_train, y_train, X_test, seed=42, epochs=45, hidden=8):
    torch.manual_seed(seed)
    np.random.seed(seed)
    in_dim = X_train.shape[1]
    kan = KAN(width=[in_dim, hidden, 2], grid=5, k=3, seed=seed, auto_save=False, device=device)
    X_tr_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_te_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    optimizer = optim.Adam(kan.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        kan.train()
        optimizer.zero_grad()
        out = kan(X_tr_t)
        loss = criterion(out, y_tr_t)
        loss.backward()
        optimizer.step()
    kan.eval()
    with torch.no_grad():
        probs_test = torch.softmax(kan(X_te_t), dim=1).cpu().numpy()[:, 1]
    return probs_test

def eval_binary_metrics(y_true, probs_pos, threshold=0.5):
    preds = (probs_pos >= threshold).astype(int)
    acc = accuracy_score(y_true, preds)
    sens = recall_score(y_true, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bacc = 0.5 * (sens + spec)
    prec = precision_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    mcc = matthews_corrcoef(y_true, preds)
    try:
        roc_auc = roc_auc_score(y_true, probs_pos)
        pr_auc = average_precision_score(y_true, probs_pos)
    except Exception:
        roc_auc, pr_auc = 0.5, 0.5
    brier = brier_score_loss(y_true, probs_pos)
    return {
        'acc': float(acc * 100),
        'sens': float(sens * 100),
        'spec': float(spec * 100),
        'bacc': float(bacc * 100),
        'prec': float(prec * 100),
        'f1': float(f1 * 100),
        'mcc': float(mcc),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'brier': float(brier)
    }

def find_youden_threshold(y_train, probs_train):
    fpr, tpr, thresholds = roc_curve(y_train, probs_train)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = float(thresholds[best_idx])
    return float(np.clip(best_thresh, 0.15, 0.90))

def compute_subject_level_metrics(y_true, probs_pos, groups, threshold=0.5):
    df_sub = pd.DataFrame({'y': y_true, 'prob': probs_pos, 'group': groups})
    grouped = df_sub.groupby('group').agg({'y': 'first', 'prob': 'mean'}).reset_index()
    return eval_binary_metrics(grouped['y'].values, grouped['prob'].values, threshold=threshold), grouped

def mcnemar_exact_p(y_true, preds_a, preds_b):
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n_disc = b + c
    if n_disc == 0:
        return 1.0, b, c
    res = binomtest(min(b, c), n=n_disc, p=0.5, alternative='two-sided')
    return float(res.pvalue), b, c

def clustered_mcnemar_test(y_true, preds_a, preds_b, groups):
    unique_g = np.unique(groups)
    cluster_diffs = []
    cluster_vars = []
    for g in unique_g:
        idx = (groups == g)
        ca = (preds_a[idx] == y_true[idx]).astype(float)
        cb = (preds_b[idx] == y_true[idx]).astype(float)
        d_i = ca - cb
        cluster_diffs.append(np.sum(d_i))
        cluster_vars.append(np.sum(d_i) ** 2)
    num = (np.sum(cluster_diffs)) ** 2
    den = np.sum(cluster_vars)
    chi2_stat = float(num / den) if den > 0 else 0.0
    p_val = float(1.0 - chi2.cdf(chi2_stat, df=1))
    return chi2_stat, p_val

def run_corpus_evaluation(X_raw, y, groups, corpus_name, k_features=None, optimal_alpha=0.60, lgb_params_strong=None):
    print(f"\n=======================================================")
    print(f"Running 5-Seed x 5-Fold GroupKFold on {corpus_name}")
    print(f"Samples: {len(y)}, Subjects: {len(np.unique(groups))}, Features: {X_raw.shape[1]}")
    print(f"=======================================================")

    model_names = [
        'LGBM_005', 'LGBM_010', 'KAN', 'MLP', 'SVM',
        'LGBM_MLP', 'LGBM_SVM', 'Stacking',
        'Fusion_Nested', 'Fusion_Optimal',
        'Fusion_Calibrated', 'Fusion_Subject_Level'
    ]
    seed_summaries = {m: [] for m in model_names}
    all_25_fold_accs = {m: [] for m in model_names}
    leakage_recordwise_accs = []
    leakage_patientwise_accs = []
    seed42_oof = {}

    for s_idx, seed in enumerate(SEEDS):
        print(f"  [{corpus_name}] Seed {s_idx+1}/5 (seed={seed})...")
        if seed == 42:
            groups_perm = groups.copy()
        else:
            rng = np.random.RandomState(seed)
            uniq_g = np.unique(groups)
            perm_map = {g: f"P{i:04d}_{g}" for i, g in enumerate(rng.permutation(uniq_g))}
            groups_perm = np.array([perm_map[g] for g in groups])

        gkf = GroupKFold(n_splits=5)
        oof_probs = {m: np.zeros(len(y)) for m in [
            'LGBM_005', 'LGBM_010', 'KAN', 'MLP', 'SVM',
            'LGBM_MLP', 'LGBM_SVM', 'Stacking',
            'Fusion_Nested', 'Fusion_Optimal'
        ]}
        oof_cal_preds = np.zeros(len(y), dtype=int)
        fold_thresholds = []
        fold_alphas = []
        fold_splits = list(gkf.split(X_raw, y, groups=groups_perm))

        # Pass 1: Train base learners on each fold
        for fold, (tr_idx, te_idx) in enumerate(fold_splits):
            X_tr, X_te = X_raw[tr_idx], X_raw[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            if k_features is not None and k_features < X_raw.shape[1]:
                sel = SelectKBest(f_classif, k=k_features)
                X_tr = sel.fit_transform(X_tr, y_tr)
                X_te = sel.transform(X_te)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            # 1. LightGBM (eta = 0.05)
            clf_005 = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=seed + fold, verbose=-1)
            clf_005.fit(X_tr_s, y_tr)
            p_lgb_005 = clf_005.predict_proba(X_te_s)[:, 1]

            # 2. LightGBM (eta = 0.10 or tuned strong)
            if lgb_params_strong is not None:
                clf_010 = lgb.LGBMClassifier(**lgb_params_strong, random_state=seed + fold, verbose=-1)
            else:
                clf_010 = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.10, random_state=seed + fold, verbose=-1)
            clf_010.fit(X_tr_s, y_tr)
            p_lgb_010 = clf_010.predict_proba(X_te_s)[:, 1]

            # 3. KAN
            p_kan = train_kan_probs(X_tr_s, y_tr, X_te_s, seed=seed + fold, epochs=45, hidden=8)

            # 4. MLP
            mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=150, random_state=seed + fold)
            mlp.fit(X_tr_s, y_tr)
            p_mlp = mlp.predict_proba(X_te_s)[:, 1]

            # 5. SVM (RBF)
            svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=seed + fold)
            svm.fit(X_tr_s, y_tr)
            p_svm = svm.predict_proba(X_te_s)[:, 1]

            oof_probs['LGBM_005'][te_idx] = p_lgb_005
            oof_probs['LGBM_010'][te_idx] = p_lgb_010
            oof_probs['KAN'][te_idx] = p_kan
            oof_probs['MLP'][te_idx] = p_mlp
            oof_probs['SVM'][te_idx] = p_svm

        # Pass 2: Leak-free out-of-fold ensemble fusion, stacking, and threshold calibration
        for fold, (tr_idx, te_idx) in enumerate(fold_splits):
            y_tr, y_te = y[tr_idx], y[te_idx]

            p_lgb_tr = oof_probs['LGBM_010'][tr_idx]
            p_kan_tr = oof_probs['KAN'][tr_idx]
            p_lgb_te = oof_probs['LGBM_010'][te_idx]
            p_kan_te = oof_probs['KAN'][te_idx]
            p_mlp_te = oof_probs['MLP'][te_idx]
            p_svm_te = oof_probs['SVM'][te_idx]

            # Ensemble partner ablations
            oof_probs['LGBM_MLP'][te_idx] = optimal_alpha * p_lgb_te + (1 - optimal_alpha) * p_mlp_te
            oof_probs['LGBM_SVM'][te_idx] = optimal_alpha * p_lgb_te + (1 - optimal_alpha) * p_svm_te
            oof_probs['Fusion_Optimal'][te_idx] = optimal_alpha * p_lgb_te + (1 - optimal_alpha) * p_kan_te

            # Out-of-fold Stacking (Logistic Regression Meta-Learner fitted strictly on tr_idx OOF)
            meta_lr = LogisticRegression()
            meta_lr.fit(np.column_stack([p_lgb_tr, p_kan_tr]), y_tr)
            oof_probs['Stacking'][te_idx] = meta_lr.predict_proba(np.column_stack([p_lgb_te, p_kan_te]))[:, 1]

            # Nested alpha selection strictly on tr_idx OOF
            best_a, best_acc = 0.5, -1.0
            for a_cand in [0.3, 0.4, 0.5, 0.6, 0.7]:
                p_c = a_cand * p_lgb_tr + (1 - a_cand) * p_kan_tr
                acc_c = accuracy_score(y_tr, (p_c >= 0.5).astype(int))
                if acc_c > best_acc:
                    best_acc = acc_c
                    best_a = a_cand
            fold_alphas.append(best_a)
            oof_probs['Fusion_Nested'][te_idx] = best_a * p_lgb_te + (1 - best_a) * p_kan_te

            # Youden's J threshold calibration strictly on tr_idx OOF
            p_fus_tr = optimal_alpha * p_lgb_tr + (1 - optimal_alpha) * p_kan_tr
            tau_star = find_youden_threshold(y_tr, p_fus_tr)
            fold_thresholds.append(tau_star)
            oof_cal_preds[te_idx] = (oof_probs['Fusion_Optimal'][te_idx] >= tau_star).astype(int)

            for m_key in oof_probs:
                f_acc = accuracy_score(y_te, (oof_probs[m_key][te_idx] >= 0.5).astype(int)) * 100
                all_25_fold_accs[m_key].append(float(f_acc))

        for m_key in oof_probs:
            met = eval_binary_metrics(y, oof_probs[m_key], threshold=0.5)
            seed_summaries[m_key].append(met)

        tn, fp, fn, tp = confusion_matrix(y, oof_cal_preds, labels=[0, 1]).ravel()
        sens_c = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec_c = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        cal_met = {
            'acc': float(accuracy_score(y, oof_cal_preds) * 100),
            'sens': float(sens_c * 100),
            'spec': float(spec_c * 100),
            'bacc': float(0.5 * (sens_c + spec_c) * 100),
            'prec': float(precision_score(y, oof_cal_preds, zero_division=0) * 100),
            'f1': float(f1_score(y, oof_cal_preds, zero_division=0) * 100),
            'mcc': float(matthews_corrcoef(y, oof_cal_preds)),
            'roc_auc': float(roc_auc_score(y, oof_probs['Fusion_Optimal'])),
            'pr_auc': float(average_precision_score(y, oof_probs['Fusion_Optimal'])),
            'tau_mean': float(np.mean(fold_thresholds))
        }
        seed_summaries['Fusion_Calibrated'].append(cal_met)

        sub_met, _ = compute_subject_level_metrics(y, oof_probs['Fusion_Optimal'], groups, threshold=0.5)
        seed_summaries['Fusion_Subject_Level'].append(sub_met)

        # Record-wise vs Patient-wise leakage test
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        rw_preds = np.zeros(len(y))
        for r_tr, r_te in skf.split(X_raw, y):
            X_rtr, X_rte = X_raw[r_tr], X_raw[r_te]
            if k_features is not None and k_features < X_raw.shape[1]:
                s_rw = SelectKBest(f_classif, k=k_features)
                X_rtr = s_rw.fit_transform(X_rtr, y[r_tr])
                X_rte = s_rw.transform(X_rte)
            sc_rw = StandardScaler()
            X_rtr = sc_rw.fit_transform(X_rtr)
            X_rte = sc_rw.transform(X_rte)
            clf_rw = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.10, random_state=seed, verbose=-1)
            clf_rw.fit(X_rtr, y[r_tr])
            rw_preds[r_te] = clf_rw.predict(X_rte)
        leakage_recordwise_accs.append(float(accuracy_score(y, rw_preds) * 100))
        leakage_patientwise_accs.append(float(seed_summaries['LGBM_010'][-1]['acc']))

        if seed == 42:
            seed42_oof = {k: v.copy() for k, v in oof_probs.items()}
            seed42_oof['y_true'] = y.copy()
            seed42_oof['groups'] = groups.copy()
            seed42_oof['cal_preds'] = oof_cal_preds.copy()
            np.savez(f"results/revision_oof_{corpus_name}.npz", **seed42_oof)

    agg = {}
    for m_key, m_list in seed_summaries.items():
        agg[m_key] = {}
        for metric_k in m_list[0].keys():
            vals = [x[metric_k] for x in m_list]
            agg[m_key][f"{metric_k}_mean"] = float(np.mean(vals))
            agg[m_key][f"{metric_k}_std"] = float(np.std(vals))
            agg[m_key][f"{metric_k}_seed42"] = float(vals[0])

    y42 = seed42_oof['y_true']
    g42 = seed42_oof['groups']
    preds_fus = (seed42_oof['Fusion_Optimal'] >= 0.5).astype(int)
    preds_lgb010 = (seed42_oof['LGBM_010'] >= 0.5).astype(int)
    preds_lgb005 = (seed42_oof['LGBM_005'] >= 0.5).astype(int)
    preds_kan = (seed42_oof['KAN'] >= 0.5).astype(int)
    preds_mlp_ens = (seed42_oof['LGBM_MLP'] >= 0.5).astype(int)
    preds_svm_ens = (seed42_oof['LGBM_SVM'] >= 0.5).astype(int)

    df_s42 = pd.DataFrame({
        'g': g42, 'y': y42,
        'p_fus': seed42_oof['Fusion_Optimal'],
        'p_lgb010': seed42_oof['LGBM_010'],
        'p_lgb005': seed42_oof['LGBM_005'],
        'p_kan': seed42_oof['KAN'],
        'p_mlp': seed42_oof['LGBM_MLP'],
        'p_svm': seed42_oof['LGBM_SVM']
    }).groupby('g').mean().reset_index()

    sub_y = df_s42['y'].round().astype(int).values
    sub_fus = (df_s42['p_fus'].values >= 0.5).astype(int)
    sub_lgb010 = (df_s42['p_lgb010'].values >= 0.5).astype(int)
    sub_lgb005 = (df_s42['p_lgb005'].values >= 0.5).astype(int)
    sub_kan = (df_s42['p_kan'].values >= 0.5).astype(int)

    stats_tests = {}
    for comp_name, p_rec, p_sub, fold25_key in [
        ('vs_LGBM_010', preds_lgb010, sub_lgb010, 'LGBM_010'),
        ('vs_LGBM_005', preds_lgb005, sub_lgb005, 'LGBM_005'),
        ('vs_KAN', preds_kan, sub_kan, 'KAN'),
        ('vs_LGBM_MLP', preds_mlp_ens, (df_s42['p_mlp'].values >= 0.5).astype(int), 'LGBM_MLP'),
        ('vs_LGBM_SVM', preds_svm_ens, (df_s42['p_svm'].values >= 0.5).astype(int), 'LGBM_SVM'),
    ]:
        rec_p, b_r, c_r = mcnemar_exact_p(y42, preds_fus, p_rec)
        clus_chi2, clus_p = clustered_mcnemar_test(y42, preds_fus, p_rec, g42)
        sub_p, b_s, c_s = mcnemar_exact_p(sub_y, sub_fus, p_sub)
        try:
            w_stat, w_p25 = wilcoxon(all_25_fold_accs['Fusion_Optimal'], all_25_fold_accs[fold25_key])
        except Exception:
            w_stat, w_p25 = 0.0, 1.0
        stats_tests[comp_name] = {
            'recording_mcnemar_p': rec_p, 'rec_b': b_r, 'rec_c': c_r,
            'clustered_obuchowski_chi2': clus_chi2, 'clustered_obuchowski_p': clus_p,
            'subject_level_mcnemar_p': sub_p, 'sub_b': b_s, 'sub_c': c_s,
            'wilcoxon_25fold_p': float(w_p25)
        }

    agg['leakage_ablation'] = {
        'recordwise_mean': float(np.mean(leakage_recordwise_accs)),
        'recordwise_std': float(np.std(leakage_recordwise_accs)),
        'patientwise_mean': float(np.mean(leakage_patientwise_accs)),
        'patientwise_std': float(np.std(leakage_patientwise_accs)),
        'inflation_delta_mean': float(np.mean(np.array(leakage_recordwise_accs) - np.array(leakage_patientwise_accs))),
        'inflation_delta_seed42': float(leakage_recordwise_accs[0] - leakage_patientwise_accs[0])
    }
    agg['statistical_tests'] = stats_tests

    print(f"[{corpus_name}] 5-Seed Summary (Seed 42 / 5-Seed Mean ± SD):")
    for m_k in ['LGBM_005', 'LGBM_010', 'KAN', 'MLP', 'SVM', 'LGBM_MLP', 'LGBM_SVM', 'Stacking', 'Fusion_Nested', 'Fusion_Optimal', 'Fusion_Calibrated', 'Fusion_Subject_Level']:
        d = agg[m_k]
        print(f"  {m_k:22s}: Seed42 Acc={d['acc_seed42']:.2f}% | 5-Seed Acc={d['acc_mean']:.2f}±{d['acc_std']:.2f}% | Spec={d['spec_mean']:.2f}% | BAcc={d['bacc_mean']:.2f}% | AUC={d['roc_auc_mean']:.4f} | PR-AUC={d['pr_auc_mean']:.4f}")
    print(f"  Leakage Inflation: Seed42 = +{agg['leakage_ablation']['inflation_delta_seed42']:.2f}%, 5-Seed Mean = +{agg['leakage_ablation']['inflation_delta_mean']:.2f}%")
    return agg

def main():
    all_results = {}

    df_tr = pd.read_csv('pd_speech_features.csv', header=1)
    y_tr = df_tr['class'].values
    g_tr = df_tr['id'].astype(str).values
    X_tr = df_tr.drop(columns=['class', 'id']).values
    all_results['turkish'] = run_corpus_evaluation(X_tr, y_tr, g_tr, 'turkish', k_features=50, optimal_alpha=0.60)

    df_bn = pd.read_csv('BenSParX/BenSParX.csv')
    y_bn = df_bn['class'].values
    g_bn = df_bn['id'].astype(str).apply(lambda x: x.split('_')[0]).values
    X_bn = df_bn.drop(columns=['class', 'id']).values
    all_results['bengali'] = run_corpus_evaluation(X_bn, y_bn, g_bn, 'bengali', k_features=None, optimal_alpha=0.50)

    df_ox = pd.read_csv('oxford_parkinsons.csv')
    y_ox = df_ox['status'].values
    g_ox = df_ox['name'].astype(str).apply(lambda x: '_'.join(x.split('_')[:3])).values
    X_ox = df_ox.drop(columns=['status', 'name']).values
    ox_strong_params = {'n_estimators': 60, 'learning_rate': 0.03, 'num_leaves': 7, 'min_child_samples': 5}
    all_results['oxford'] = run_corpus_evaluation(X_ox, y_ox, g_ox, 'oxford', k_features=None, optimal_alpha=0.50, lgb_params_strong=ox_strong_params)

    with open('results/revision_multiseed_ablation.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved results/revision_multiseed_ablation.json")

if __name__ == '__main__':
    main()
