import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
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
        logits = kan(X_test_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs

def main():
    print("Loading Turkish Dataset for Feature Selection...")
    df = pd.read_csv('pd_speech_features.csv', header=1)
    y = df['class'].values
    groups = df['id'].values
    X_df = df.drop(columns=['class', 'id'])
    
    k_values = [50, 100, 200, 300, 500]
    
    for k in k_values:
        print(f"\n======================================")
        print(f"Testing with Top {k} Features (ANOVA)")
        print(f"======================================")
        
        gkf = GroupKFold(n_splits=5)
        lgb_accs, kan_accs, fusion_accs = [], [], []
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X_df, y, groups=groups)):
            X_train, X_test = X_df.iloc[train_idx].values, X_df.iloc[test_idx].values
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 1. Feature Selection (Fitted ONLY on training data to prevent leakage)
            selector = SelectKBest(f_classif, k=k)
            X_train_sel = selector.fit_transform(X_train, y_train)
            X_test_sel = selector.transform(X_test)
            
            # 2. Scaling
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train_sel)
            X_test_s = scaler.transform(X_test_sel)
            
            # LightGBM
            lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1, n_estimators=100)
            lgb_model.fit(X_train_s, y_train)
            probs_lgb = lgb_model.predict_proba(X_test_s)
            preds_lgb = np.argmax(probs_lgb, axis=1)
            lgb_accs.append(accuracy_score(y_test, preds_lgb))
            
            # KAN
            X_train_t = torch.tensor(X_train_s, dtype=torch.float32).to(device)
            y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
            X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)
            probs_kan = train_kan_fold(X_train_t, y_train_t, X_test_t, k, seed=42+fold)
            preds_kan = np.argmax(probs_kan, axis=1)
            kan_accs.append(accuracy_score(y_test, preds_kan))
            
            # Fusion
            probs_fused = (probs_lgb + probs_kan) / 2.0
            preds_fused = np.argmax(probs_fused, axis=1)
            fusion_accs.append(accuracy_score(y_test, preds_fused))
            
        print(f"LightGBM Mean: {np.mean(lgb_accs)*100:.2f}%")
        print(f"KAN Mean:      {np.mean(kan_accs)*100:.2f}%")
        print(f"Fusion Mean:   {np.mean(fusion_accs)*100:.2f}%")

if __name__ == '__main__':
    main()
