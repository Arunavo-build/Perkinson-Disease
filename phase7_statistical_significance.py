import os
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)

def mcnemar_exact_test(table):
    """Computes McNemar test using scipy.stats"""
    # table = [[n11, n10], [n01, n00]]
    b = table[0][1] # proposed correct, baseline wrong
    c = table[1][0] # baseline correct, proposed wrong
    n_disc = b + c
    if n_disc == 0:
        return 0.0, 1.0
    if n_disc < 25:
        # Exact binomial test
        res = stats.binomtest(b, n_disc, p=0.5, alternative='two-sided')
        return float(b), res.pvalue
    else:
        # Chi-squared with Edwards continuity correction
        chi2_stat = ((abs(b - c) - 1.0) ** 2) / float(n_disc)
        pval = 1.0 - stats.chi2.cdf(chi2_stat, df=1)
        return chi2_stat, pval

def wilson_score_interval(correct, total, confidence=0.95):
    """Calculates Wilson score confidence interval for a proportion."""
    if total == 0:
        return 0, 0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = correct / total
    denom = 1 + (z**2) / total
    center = (p + (z**2) / (2 * total)) / denom
    margin = (z * np.sqrt((p * (1 - p) / total) + (z**2) / (4 * (total**2)))) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower, upper

def run_significance_tests():
    print("=== STARTING PHASE 7: STATISTICAL SIGNIFICANCE TESTING ===")
    
    datasets = ['turkish_uci', 'bengali_bensparx', 'oxford_uci']
    test_results = []
    
    for ds in datasets:
        pred_file = f"results/predictions_{ds}.npz"
        if not os.path.exists(pred_file):
            print(f"Warning: {pred_file} not found yet. Running on available data.")
            continue
            
        data = np.load(pred_file)
        y_true = data['y_true']
        oof_lgb = data['oof_lgb']
        oof_kan = data['oof_kan']
        oof_naive = data['oof_naive']
        oof_adapt = data['oof_adapt']
        
        pred_lgb = np.argmax(oof_lgb, axis=1)
        pred_kan = np.argmax(oof_kan, axis=1)
        pred_naive = np.argmax(oof_naive, axis=1)
        pred_adapt = np.argmax(oof_adapt, axis=1)
        
        corr_lgb = (pred_lgb == y_true)
        corr_kan = (pred_kan == y_true)
        corr_naive = (pred_naive == y_true)
        corr_adapt = (pred_adapt == y_true)
        
        n_total = len(y_true)
        acc_adapt = np.mean(corr_adapt)
        ci_low, ci_high = wilson_score_interval(np.sum(corr_adapt), n_total)
        
        print(f"\n--- Statistical Testing on {ds.upper()} (N={n_total}) ---")
        print(f"Adaptive Fusion Accuracy: {acc_adapt*100:.2f}% (95% CI: [{ci_low*100:.2f}%, {ci_high*100:.2f}%])")
        
        comparisons = [
            ('Adaptive_Fusion vs LightGBM', corr_adapt, corr_lgb, pred_adapt, pred_lgb),
            ('Adaptive_Fusion vs KAN', corr_adapt, corr_kan, pred_adapt, pred_kan),
            ('Adaptive_Fusion vs Naive_Fusion', corr_adapt, corr_naive, pred_adapt, pred_naive),
        ]
        
        for name, c_prop, c_base, p_prop, p_base in comparisons:
            # Contingency table
            # [[both correct, prop correct & base wrong],
            #  [prop wrong & base correct, both wrong]]
            n11 = np.sum(c_prop & c_base)
            n10 = np.sum(c_prop & (~c_base)) # b: proposed wins
            n01 = np.sum((~c_prop) & c_base) # c: baseline wins
            n00 = np.sum((~c_prop) & (~c_base))
            
            table = [[n11, n10], [n01, n00]]
            
            # McNemar Test
            stat, pval = mcnemar_exact_test(table)
            
            # Wilcoxon signed rank test on fold/sample differences
            diff = c_prop.astype(int) - c_base.astype(int)
            if np.any(diff != 0):
                w_stat, w_pval = stats.wilcoxon(diff[diff != 0])
            else:
                w_stat, w_pval = 0.0, 1.0
                
            print(f"  {name:<32}: b={n10}, c={n01} | McNemar p={pval:.4f} | Wilcoxon p={w_pval:.4f} | Sig: {'YES (p < 0.05)' if pval < 0.05 or w_pval < 0.05 else 'Trend'}")
            
            test_results.append({
                'dataset': ds,
                'comparison': name,
                'proposed_correct': np.sum(c_prop),
                'baseline_correct': np.sum(c_base),
                'discordant_proposed_wins (b)': n10,
                'discordant_baseline_wins (c)': n01,
                'mcnemar_statistic': stat,
                'mcnemar_pvalue': pval,
                'wilcoxon_statistic': w_stat,
                'wilcoxon_pvalue': w_pval,
                'statistically_significant': (pval < 0.05) or (w_pval < 0.05),
                'ci_95_lower': ci_low,
                'ci_95_upper': ci_high
            })
            
    res_df = pd.DataFrame(test_results)
    res_df.to_csv('results/phase7_statistical_tests.csv', index=False)
    print("\nPhase 7 Statistical Significance Testing completed and saved to results/phase7_statistical_tests.csv!")

if __name__ == '__main__':
    run_significance_tests()
