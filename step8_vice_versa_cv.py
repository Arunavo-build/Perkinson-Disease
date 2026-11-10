import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from kan import KAN
import warnings
warnings.filterwarnings('ignore')

def train_and_test(X_train, y_train, X_test, y_test, name):
    print(f"\n======================================")
    print(f"Running: {name}")
    print(f"======================================")
    
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train)
    X_test_np = scaler.transform(X_test)
    
    print("Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    lgb_model.fit(X_train_np, y_train)
    probs_lgb = lgb_model.predict_proba(X_test_np)
    preds_lgb = lgb_model.predict(X_test_np)
    acc_lgb = accuracy_score(y_test, preds_lgb)
    print(f"LightGBM Accuracy: {acc_lgb:.4f}")
    
    print("Training KAN...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_t = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    
    input_dim = X_train_np.shape[1]
    kan_model = KAN(width=[input_dim, 10, 2], grid=5, k=3, seed=42, device=device)
    
    optimizer = optim.Adam(kan_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 100
    for epoch in range(epochs):
        kan_model.train()
        optimizer.zero_grad()
        output = kan_model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()
            
    kan_model.eval()
    with torch.no_grad():
        logits_kan = kan_model(X_test_t)
        probs_kan_t = torch.softmax(logits_kan, dim=1).cpu().numpy()
        preds_kan = np.argmax(probs_kan_t, axis=1)
    acc_kan = accuracy_score(y_test, preds_kan)
    print(f"KAN Accuracy: {acc_kan:.4f}")
    
    # Soft Voting
    fused_probs = (probs_lgb + probs_kan_t) / 2
    fused_preds = np.argmax(fused_probs, axis=1)
    
    acc_fusion = accuracy_score(y_test, fused_preds)
    print(f"Fusion Accuracy: {acc_fusion:.4f}")
    return acc_lgb, acc_kan, acc_fusion

def main():
    print("Loading Datasets...")
    df_A = pd.read_csv('pd_speech_features.csv', header=1)
    df_B = pd.read_csv('BenSParX/BenSParX.csv')
    
    # Target columns
    y_A = df_A['class'].values
    y_B = df_B['class'].values
    
    # Find common features (excluding 'id' and 'class')
    common_cols = list(set(df_A.columns).intersection(set(df_B.columns)))
    common_features = [c for c in common_cols if c not in ['id', 'class']]
    
    print(f"Found {len(common_features)} common features.")
    
    X_A = df_A[common_features].values
    X_B = df_B[common_features].values
    
    # 1. Train on A, Test on B
    train_and_test(X_A, y_A, X_B, y_B, "Train on UCI (Dataset A), Test on BenSParX (Dataset B)")
    
    # 2. Train on B, Test on A
    train_and_test(X_B, y_B, X_A, y_A, "Train on BenSParX (Dataset B), Test on UCI (Dataset A)")

if __name__ == '__main__':
    main()
