import os
import json
import numpy as np
import pandas as pd
import scipy.linalg
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)
SEEDS = [42, 100, 2024, 777, 999]

def coral_align(Xs, Xt):
    cov_s = np.cov(Xs, rowvar=False) + np.eye(Xs.shape[1]) * 1e-4
    cov_t = np.cov(Xt, rowvar=False) + np.eye(Xt.shape[1]) * 1e-4
    A_s = scipy.linalg.inv(scipy.linalg.sqrtm(cov_s))
    A_t = scipy.linalg.sqrtm(cov_t)
    return np.real(Xs.dot(A_s).dot(A_t))

def analyze_mfcc2_and_energy_directionality(df_tr, df_bn):
    print("\n--- Analyzing Feature Directionality & Polarity Across Turkish and Bengali ---")
    direction_report = {}

    pairs = [
        ('MFCC_2', 'mean_MFCC_2nd_coef', 'MFCC_2', '2nd Mel-Frequency Cepstral Coefficient'),
        ('Delta_MFCC_2', 'mean_2nd_delta', 'Delta_MFCC_2', '1st-Order Velocity of 2nd MFCC'),
        ('Delta_MFCC_6', 'mean_6th_delta', 'Delta_MFCC_6', '1st-Order Velocity of 6th MFCC'),
        ('Log_Energy_Accel', 'std_delta_delta_log_energy', 'std_delta_delta_log_energy', 'Std. Dev. of Log-Energy Acceleration'),
        ('HNR_Harmonicity', 'meanHarmToNoiseHarmonicity', 'HNR', 'Harmonics-to-Noise Ratio (dB)'),
        ('AutoCorr_Harm', 'meanAutoCorrHarmonicity', 'AutoCorrHarmonicity', 'Autocorrelation Harmonicity'),
        ('Local_Jitter', 'locPctJitter', 'locPctJitter', 'Local Cycle-to-Cycle Pitch Jitter (%)'),
        ('Local_Shimmer', 'locShimmer', 'locShimmer', 'Local Cycle-to-Cycle Amplitude Shimmer'),
        ('Formant_F1', 'f1', 'f1', 'First Vocal Tract Formant Frequency F1 (Hz)')
    ]

    for key, col_tr, col_bn, desc in pairs:
        tr_pd = df_tr[df_tr['class'] == 1][col_tr].values
        tr_hc = df_tr[df_tr['class'] == 0][col_tr].values
        bn_pd = df_bn[df_bn['class'] == 1][col_bn].values
        bn_hc = df_bn[df_bn['class'] == 0][col_bn].values

        tr_d = (np.mean(tr_pd) - np.mean(tr_hc)) / (np.sqrt(0.5 * (np.var(tr_pd) + np.var(tr_hc))) + 1e-8)
        bn_d = (np.mean(bn_pd) - np.mean(bn_hc)) / (np.sqrt(0.5 * (np.var(bn_pd) + np.var(bn_hc))) + 1e-8)
        _, tr_p = mannwhitneyu(tr_pd, tr_hc)
        _, bn_p = mannwhitneyu(bn_pd, bn_hc)

        same_sign = bool((tr_d * bn_d) > 0)
        direction_report[key] = {
            'description': desc,
            'tr_col': col_tr,
            'bn_col': col_bn,
            'tr_pd_mean': float(np.mean(tr_pd)), 'tr_pd_sd': float(np.std(tr_pd)),
            'tr_hc_mean': float(np.mean(tr_hc)), 'tr_hc_sd': float(np.std(tr_hc)),
            'tr_cohen_d': float(tr_d), 'tr_p': float(tr_p),
            'bn_pd_mean': float(np.mean(bn_pd)), 'bn_pd_sd': float(np.std(bn_pd)),
            'bn_hc_mean': float(np.mean(bn_hc)), 'bn_hc_sd': float(np.std(bn_hc)),
            'bn_cohen_d': float(bn_d), 'bn_p': float(bn_p),
            'directional_consistency': 'Consistent Sign' if same_sign else 'Opposite Polarity (Sign Flip)'
        }
        print(f"  {key:18s} | TR: PD={np.mean(tr_pd):.3f} vs HC={np.mean(tr_hc):.3f} (d={tr_d:+.2f}) | BN: PD={np.mean(bn_pd):.3f} vs HC={np.mean(bn_hc):.3f} (d={bn_d:+.2f}) | {direction_report[key]['directional_consistency']}")

    return direction_report

def run_few_shot_learning_curves(Xs_raw, ys, Xt_raw, yt, gt, direction_name):
    print(f"\n--- Running Few-Shot Learning Curves: {direction_name} ---")
    scaler_s = StandardScaler()
    Xs_s = scaler_s.fit_transform(Xs_raw)

    # Zero-shot evaluation on full target cohort
    scaler_t_full = StandardScaler()
    Xt_full_s = scaler_t_full.fit_transform(Xt_raw)

    clf_zs = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    clf_zs.fit(Xs_s, ys)
    zs_probs = clf_zs.predict_proba(Xt_full_s)[:, 1]
    zs_acc = float(accuracy_score(yt, (zs_probs >= 0.5).astype(int)) * 100)
    zs_auc = float(roc_auc_score(yt, zs_probs))

    Xs_coral_full = coral_align(Xs_s, Xt_full_s)
    clf_coral_zs = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    clf_coral_zs.fit(Xs_coral_full, ys)
    czs_probs = clf_coral_zs.predict_proba(Xt_full_s)[:, 1]
    czs_acc = float(accuracy_score(yt, (czs_probs >= 0.5).astype(int)) * 100)
    czs_auc = float(roc_auc_score(yt, czs_probs))

    # Patient-grouped few-shot fractions: 5%, 10%, 15%, 20%, 30%, 50%
    fractions = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    unique_patients = np.unique(gt)
    pat_labels = np.array([yt[gt == p][0] for p in unique_patients])
    pos_pats = unique_patients[pat_labels == 1]
    neg_pats = unique_patients[pat_labels == 0]

    curve_results = {
        'zero_shot': {'acc': zs_acc, 'auc': zs_auc},
        'unsupervised_coral': {'acc': czs_acc, 'auc': czs_auc},
        'fractions': {}
    }

    for frac in fractions:
        target_only_accs, transfer_no_coral_accs, coral_transfer_accs = [], [], []
        target_only_aucs, transfer_no_coral_aucs, coral_transfer_aucs = [], [], []
        n_pos = max(1, int(round(len(pos_pats) * frac)))
        n_neg = max(1, int(round(len(neg_pats) * frac)))
        n_calib_pats = n_pos + n_neg

        for seed in SEEDS:
            rng = np.random.RandomState(seed + int(frac * 1000))
            cal_pos = rng.choice(pos_pats, size=n_pos, replace=False)
            cal_neg = rng.choice(neg_pats, size=n_neg, replace=False)
            cal_pats = set(np.concatenate([cal_pos, cal_neg]))

            cal_mask = np.array([p in cal_pats for p in gt])
            test_mask = ~cal_mask

            Xt_cal_raw, yt_cal = Xt_raw[cal_mask], yt[cal_mask]
            Xt_te_raw, yt_te = Xt_raw[test_mask], yt[test_mask]

            # Fit target scaler strictly on calibration set (or unlabelled target covariance for CORAL)
            sc_t = StandardScaler()
            Xt_cal_s = sc_t.fit_transform(Xt_cal_raw)
            Xt_te_s = sc_t.transform(Xt_te_raw)

            # 1. Target-Only (Trained purely on scarce target patients)
            min_c = max(2, min(10, len(yt_cal) // 3))
            clf_to = lgb.LGBMClassifier(n_estimators=80, learning_rate=0.05, min_child_samples=min_c, random_state=seed, verbose=-1)
            clf_to.fit(Xt_cal_s, yt_cal)
            p_to = clf_to.predict_proba(Xt_te_s)[:, 1]
            target_only_accs.append(float(accuracy_score(yt_te, (p_to >= 0.5).astype(int)) * 100))
            target_only_aucs.append(float(roc_auc_score(yt_te, p_to)))

            # 2. Source + Target Fine-Tuning (No CORAL; weighting target samples proportionally)
            weight_t = max(1.0, len(ys) / (len(yt_cal) * 1.5))
            sample_w = np.concatenate([np.ones(len(ys)), np.full(len(yt_cal), weight_t)])
            X_tr_nc = np.vstack([Xs_s, Xt_cal_s])
            y_tr_nc = np.concatenate([ys, yt_cal])
            clf_nc = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, min_child_samples=min_c, random_state=seed, verbose=-1)
            clf_nc.fit(X_tr_nc, y_tr_nc, sample_weight=sample_w)
            p_nc = clf_nc.predict_proba(Xt_te_s)[:, 1]
            transfer_no_coral_accs.append(float(accuracy_score(yt_te, (p_nc >= 0.5).astype(int)) * 100))
            transfer_no_coral_aucs.append(float(roc_auc_score(yt_te, p_nc)))

            # 3. CORAL + Source + Target Fine-Tuning
            Xs_c = coral_align(Xs_s, Xt_cal_s)
            X_tr_c = np.vstack([Xs_c, Xt_cal_s])
            clf_c = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, min_child_samples=min_c, random_state=seed, verbose=-1)
            clf_c.fit(X_tr_c, y_tr_nc, sample_weight=sample_w)
            p_c = clf_c.predict_proba(Xt_te_s)[:, 1]
            coral_transfer_accs.append(float(accuracy_score(yt_te, (p_c >= 0.5).astype(int)) * 100))
            coral_transfer_aucs.append(float(roc_auc_score(yt_te, p_c)))

        f_key = f"{int(frac*100)}pct"
        curve_results['fractions'][f_key] = {
            'n_target_patients': int(n_calib_pats),
            'target_only_acc_mean': float(np.mean(target_only_accs)),
            'target_only_acc_std': float(np.std(target_only_accs)),
            'target_only_auc_mean': float(np.mean(target_only_aucs)),
            'transfer_no_coral_acc_mean': float(np.mean(transfer_no_coral_accs)),
            'transfer_no_coral_acc_std': float(np.std(transfer_no_coral_accs)),
            'transfer_no_coral_auc_mean': float(np.mean(transfer_no_coral_aucs)),
            'coral_transfer_acc_mean': float(np.mean(coral_transfer_accs)),
            'coral_transfer_acc_std': float(np.std(coral_transfer_accs)),
            'coral_transfer_auc_mean': float(np.mean(coral_transfer_aucs))
        }
        d = curve_results['fractions'][f_key]
        print(f"  Budget {int(frac*100):2d}% ({n_calib_pats:2d} pats) | Target-Only: {d['target_only_acc_mean']:.2f}±{d['target_only_acc_std']:.2f}% | No-CORAL Transfer: {d['transfer_no_coral_acc_mean']:.2f}±{d['transfer_no_coral_acc_std']:.2f}% | CORAL+Transfer: {d['coral_transfer_acc_mean']:.2f}±{d['coral_transfer_acc_std']:.2f}%")

    return curve_results

def build_homologous_feature_mapping(df_tr, df_bn, df_ox):
    # Exact 27 shared features between Turkish and Bengali (plus semantic MFCC mapping)
    common_exact = [c for c in df_tr.columns if c in df_bn.columns and c not in ['id', 'class']]
    rows = []
    for idx, col in enumerate(common_exact, 1):
        if 'Jitter' in col:
            cat = 'Fundamental Frequency Perturbation (Jitter)'
        elif 'Shimmer' in col:
            cat = 'Amplitude Perturbation (Shimmer)'
        elif 'Pulses' in col or 'Voicing' in col:
            cat = 'Glottal Pulse Periodicity'
        elif 'Pitch' in col or 'Int' in col:
            cat = 'Vocal Pitch & Subglottal Intensity'
        elif 'Harmonic' in col or 'GNE' in col:
            cat = 'Glottal Harmonic-to-Noise Ratio'
        elif 'f1' in col or 'f2' in col or 'f3' in col or 'f4' in col or 'b1' in col or 'b2' in col or 'b3' in col or 'b4' in col:
            cat = 'Vocal Tract Formant Frequencies & Bandwidths'
        else:
            cat = 'Dynamic Log-Energy Kinematics'
        rows.append({
            'Index': idx,
            'Feature_Category': cat,
            'Turkish_UCI_Column': col,
            'Bengali_BenSParX_Column': col,
            'Extraction_Difference': 'Sustained /a/ (Praat, 44.1 kHz clinical mic) vs. Spontaneous Bengali dialogue (Praat/Parselmouth, smartphone mic)'
        })
    df_map = pd.DataFrame(rows)
    df_map.to_csv('results/homologous_feature_mapping.csv', index=False)
    print(f"\nSaved {len(rows)} exact homologous features to results/homologous_feature_mapping.csv")
    return rows

def main():
    df_tr = pd.read_csv('pd_speech_features.csv', header=1)
    y_tr = df_tr['class'].values
    g_tr = df_tr['id'].astype(str).values

    df_bn = pd.read_csv('BenSParX/BenSParX.csv')
    y_bn = df_bn['class'].values
    g_bn = df_bn['id'].astype(str).apply(lambda x: x.split('_')[0]).values

    df_ox = pd.read_csv('oxford_parkinsons.csv')
    y_ox = df_ox['status'].values
    g_ox = df_ox['name'].astype(str).apply(lambda x: '_'.join(x.split('_')[:3])).values

    dir_report = analyze_mfcc2_and_energy_directionality(df_tr, df_bn)
    map_rows = build_homologous_feature_mapping(df_tr, df_bn, df_ox)

    common_tr_bn = [c for c in df_tr.columns if c in df_bn.columns and c not in ['id', 'class']]
    X_tr_27 = df_tr[common_tr_bn].values
    X_bn_27 = df_bn[common_tr_bn].values

    ox_to_tr = [
        ('MDVP:Jitter(%)', 'locPctJitter'), ('MDVP:Jitter(Abs)', 'locAbsJitter'),
        ('MDVP:RAP', 'rapJitter'), ('MDVP:PPQ', 'ppq5Jitter'), ('Jitter:DDP', 'ddpJitter'),
        ('MDVP:Shimmer', 'locShimmer'), ('MDVP:Shimmer(dB)', 'locDbShimmer'),
        ('Shimmer:APQ3', 'apq3Shimmer'), ('Shimmer:APQ5', 'apq5Shimmer'),
        ('Shimmer:DDA', 'ddaShimmer'), ('RPDE', 'RPDE'), ('DFA', 'DFA'), ('PPE', 'PPE')
    ]
    tr_13_cols = [tr for ox, tr in ox_to_tr]
    ox_13_cols = [ox for ox, tr in ox_to_tr]
    X_tr_13 = df_tr[tr_13_cols].values
    X_ox_13 = df_ox[ox_13_cols].values

    curves = {
        'TR_to_BN': run_few_shot_learning_curves(X_tr_27, y_tr, X_bn_27, y_bn, g_bn, 'Turkish -> Bengali (27 Shared Features)'),
        'BN_to_TR': run_few_shot_learning_curves(X_bn_27, y_bn, X_tr_27, y_tr, g_tr, 'Bengali -> Turkish (27 Shared Features)'),
        'TR_to_OX': run_few_shot_learning_curves(X_tr_13, y_tr, X_ox_13, y_ox, g_ox, 'Turkish -> English Oxford (13 Shared Features)')
    }

    out = {
        'directionality_analysis': dir_report,
        'homologous_mapping_27': map_rows,
        'learning_curves': curves
    }
    with open('results/revision_learning_curves.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/revision_learning_curves.json")

if __name__ == '__main__':
    main()
