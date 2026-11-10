import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from kan import KAN
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_kan_fold(X_train_t, y_train_t, X_test_t, input_dim, seed=42):
    torch.manual_seed(seed)
    kan = KAN(width=[input_dim, 8, 2], grid=5, k=3, seed=seed, device=device)
    optimizer = optim.Adam(kan.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    epochs = 60
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
    return probs_test

def main():
    print("Loading Bengali Dataset (BenSParX)...")
    df = pd.read_csv('BenSParX/BenSParX.csv')
    y = df['class'].values
    
    # Crucial: Extract actual patient ID for BenSParX (e.g., 'PD1' from 'PD1_1')
    groups = df['id'].astype(str).apply(lambda x: x.split('_')[0]).values
    
    X_df = df.drop(columns=['class', 'id'])
    
    gkf = GroupKFold(n_splits=5)
    
    weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    accs_by_weight = {w: [] for w in weights}
    
    lgb_baseline_accs = []
    kan_baseline_accs = []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_df, y, groups=groups)):
        X_train, X_test = X_df.iloc[train_idx].values, X_df.iloc[test_idx].values
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # LightGBM
        lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1, n_estimators=100)
        lgb_model.fit(X_train_s, y_train)
        probs_lgb = lgb_model.predict_proba(X_test_s)
        lgb_baseline_accs.append(accuracy_score(y_test, np.argmax(probs_lgb, axis=1)))
        
        # KAN
        X_train_t = torch.tensor(X_train_s, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
        X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)
        probs_kan = train_kan_fold(X_train_t, y_train_t, X_test_t, X_train_s.shape[1], seed=42+fold)
        kan_baseline_accs.append(accuracy_score(y_test, np.argmax(probs_kan, axis=1)))
        
        # Adaptive Fusion
        for w in weights:
            probs_fused = w * probs_lgb + (1 - w) * probs_kan
            preds_fused = np.argmax(probs_fused, axis=1)
            accs_by_weight[w].append(accuracy_score(y_test, preds_fused))
            
    print("\n--- Baseline Individual Models (Patient-wise 5-Fold) ---")
    print(f"LightGBM Mean: {np.mean(lgb_baseline_accs)*100:.2f}%")
    print(f"KAN Mean:      {np.mean(kan_baseline_accs)*100:.2f}%")
    
    print("\n--- Adaptive Weighted Fusion Results ---")
    for w in weights:
        print(f'Weight LGB={w:.1f} / KAN={1-w:.1f} : Fusion Acc = {np.mean(accs_by_weight[w])*100:.2f}%')

if __name__ == '__main__':
    main()
