import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
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
        logits_train = kan(X_train_t)
        probs_train = torch.softmax(logits_train, dim=1).cpu().numpy()
        
        logits_test = kan(X_test_t)
        probs_test = torch.softmax(logits_test, dim=1).cpu().numpy()
    return probs_train, probs_test

def main():
    print("Loading Turkish Dataset for Meta-Classifier Stacking...")
    df = pd.read_csv('pd_speech_features.csv', header=1)
    y = df['class'].values
    groups = df['id'].values
    X_df = df.drop(columns=['class', 'id'])
    
    gkf = GroupKFold(n_splits=5)
    lgb_accs, kan_accs, soft_fusion_accs, stacking_accs = [], [], [], []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_df, y, groups=groups)):
        print(f"Fold {fold+1}/5")
        X_train, X_test = X_df.iloc[train_idx].values, X_df.iloc[test_idx].values
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scaling
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # 1. LightGBM
        lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1, n_estimators=100)
        lgb_model.fit(X_train_s, y_train)
        probs_train_lgb = lgb_model.predict_proba(X_train_s)
        probs_test_lgb = lgb_model.predict_proba(X_test_s)
        
        preds_lgb = np.argmax(probs_test_lgb, axis=1)
        lgb_accs.append(accuracy_score(y_test, preds_lgb))
        
        # 2. KAN
        X_train_t = torch.tensor(X_train_s, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
        X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)
        probs_train_kan, probs_test_kan = train_kan_fold(X_train_t, y_train_t, X_test_t, X_train_s.shape[1], seed=42+fold)
        
        preds_kan = np.argmax(probs_test_kan, axis=1)
        kan_accs.append(accuracy_score(y_test, preds_kan))
        
        # 3. Soft Voting Fusion
        probs_fused = (probs_test_lgb + probs_test_kan) / 2.0
        preds_fused = np.argmax(probs_fused, axis=1)
        soft_fusion_accs.append(accuracy_score(y_test, preds_fused))
        
        # 4. Meta-Classifier Stacking
        # Features for meta-model: Probabilities of class 1 from LGBM and KAN
        meta_X_train = np.column_stack((probs_train_lgb[:, 1], probs_train_kan[:, 1]))
        meta_X_test = np.column_stack((probs_test_lgb[:, 1], probs_test_kan[:, 1]))
        
        meta_model = LogisticRegression()
        meta_model.fit(meta_X_train, y_train)
        preds_stack = meta_model.predict(meta_X_test)
        stacking_accs.append(accuracy_score(y_test, preds_stack))
        
    print(f"\n--- Final Results (Patient-wise 5-Fold) ---")
    print(f"LightGBM Mean:            {np.mean(lgb_accs)*100:.2f}%")
    print(f"KAN Mean:                 {np.mean(kan_accs)*100:.2f}%")
    print(f"Naive Soft Fusion Mean:   {np.mean(soft_fusion_accs)*100:.2f}%")
    print(f"Meta-Stacking Fusion:     {np.mean(stacking_accs)*100:.2f}%")

if __name__ == '__main__':
    main()
