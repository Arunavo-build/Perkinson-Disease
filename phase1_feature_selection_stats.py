import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import f_classif, mutual_info_classif
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

def fdr_correction(p_values):
    """Benjamini-Hochberg False Discovery Rate (FDR) correction"""
    p_values = np.asarray(p_values)
    n = len(p_values)
    order = np.argsort(p_values)
    ranked_p = p_values[order]
    q_values = np.zeros(n)
    running_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        q = (ranked_p[i] * n) / rank
        running_min = min(running_min, q)
        q_values[order[i]] = running_min
    return np.clip(q_values, 0, 1)

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def analyze_dataset_features(df, target_col, dataset_name, exclude_cols=['id', 'name']):
    print(f"\n=======================================================")
    print(f"Phase 1: Statistical Acoustic Profiling on {dataset_name}")
    print(f"=======================================================")
    
    feature_cols = [c for c in df.columns if c != target_col and c not in exclude_cols]
    y = df[target_col].values
    X = df[feature_cols].copy()
    
    # Handle NaNs if any
    X = X.fillna(X.median())
    
    pd_mask = (y == 1)
    hc_mask = (y == 0)
    
    results = []
    
    # 1. Classical Statistical Tests
    t_pvals = []
    u_pvals = []
    for col in feature_cols:
        pd_vals = X.loc[pd_mask, col].values
        hc_vals = X.loc[hc_mask, col].values
        
        # Welch's t-test
        t_stat, t_p = stats.ttest_ind(pd_vals, hc_vals, equal_var=False)
        # Mann-Whitney U test
        u_stat, u_p = stats.mannwhitneyu(pd_vals, hc_vals, alternative='two-sided')
        d = cohens_d(pd_vals, hc_vals)
        
        t_pvals.append(t_p if not np.isnan(t_p) else 1.0)
        u_pvals.append(u_p if not np.isnan(u_p) else 1.0)
        
        results.append({
            'dataset': dataset_name,
            'feature': col,
            'pd_mean': np.mean(pd_vals),
            'pd_std': np.std(pd_vals),
            'hc_mean': np.mean(hc_vals),
            'hc_std': np.std(hc_vals),
            't_statistic': t_stat,
            't_pvalue': t_p,
            'u_statistic': u_stat,
            'u_pvalue': u_p,
            'cohens_d': d
        })
        
    res_df = pd.DataFrame(results)
    res_df['t_qvalue_fdr'] = fdr_correction(t_pvals)
    res_df['u_qvalue_fdr'] = fdr_correction(u_pvals)
    
    # 2. ANOVA F-Score
    f_scores, f_pvals = f_classif(X, y)
    res_df['anova_f_score'] = f_scores
    res_df['anova_f_pval'] = f_pvals
    
    # 3. Mutual Information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    res_df['mutual_info'] = mi_scores
    
    # 4. LightGBM Tree Feature Importance
    clf = lgb.LGBMClassifier(random_state=42, verbose=-1, n_estimators=100)
    clf.fit(X.values, y)
    res_df['tree_importance'] = clf.feature_importances_
    
    # Composite Rank Score
    res_df['rank_f'] = res_df['anova_f_score'].rank(ascending=False)
    res_df['rank_mi'] = res_df['mutual_info'].rank(ascending=False)
    res_df['rank_tree'] = res_df['tree_importance'].rank(ascending=False)
    res_df['composite_rank'] = (res_df['rank_f'] + res_df['rank_mi'] + res_df['rank_tree']) / 3.0
    
    res_df = res_df.sort_values(by='composite_rank').reset_index(drop=True)
    
    output_path = f"results/{dataset_name.lower().replace(' ', '_')}_feature_stats.csv"
    res_df.to_csv(output_path, index=False)
    print(f"Saved complete feature statistics to {output_path}")
    
    sig_count = (res_df['t_qvalue_fdr'] < 0.05).sum()
    print(f"Total features: {len(feature_cols)}")
    print(f"Significantly differentiated features (FDR q < 0.05): {sig_count} ({(sig_count/len(feature_cols))*100:.1f}%)")
    print(f"Top 5 Discriminatory Features:")
    for idx, row in res_df.head(5).iterrows():
        print(f"  {idx+1}. {row['feature']} | Cohen's d: {row['cohens_d']:.3f} | ANOVA F: {row['anova_f_score']:.2f} | Tree Imp: {row['tree_importance']}")
        
    return res_df

def main():
    print("=== STARTING PHASE 1: STATISTICAL & FEATURE SELECTION ANALYSIS ===")
    
    # 1. Turkish UCI Dataset
    df_turkish = pd.read_csv('pd_speech_features.csv', header=1)
    analyze_dataset_features(df_turkish, target_col='class', dataset_name='Turkish_UCI', exclude_cols=['id'])
    
    # 2. Bengali BenSParX Dataset
    df_bengali = pd.read_csv('BenSParX/BenSParX.csv')
    analyze_dataset_features(df_bengali, target_col='class', dataset_name='Bengali_BenSParX', exclude_cols=['id'])
    
    # 3. Oxford UCI Dataset
    df_oxford = pd.read_csv('oxford_parkinsons.csv')
    analyze_dataset_features(df_oxford, target_col='status', dataset_name='Oxford_UCI', exclude_cols=['name', 'subject_id'])
    
    print("\nPhase 1 execution completed successfully!")

if __name__ == '__main__':
    main()
