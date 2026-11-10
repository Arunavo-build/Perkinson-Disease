import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb
from kan import KAN
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Loading BenSParX Dataset...")
    df = pd.read_csv('BenSParX/BenSParX.csv')
    
    # Assuming 'class' or 'status' or 'target' is the label
    target_col = [col for col in df.columns if col.lower() in ['class', 'status', 'target', 'label', 'pd']][-1]
    y = df[target_col].values
    X_df = df.drop(columns=[target_col])
    
    id_col = 'id' if 'id' in X_df.columns else None
    if id_col:
        ids = X_df[id_col].values
        X_df = X_df.drop(columns=[id_col])
    else:
        ids = np.arange(len(y))
        
    # We use a standard train/test split. Grouped split could also be used if ID is present.
    # For a quick test, let's do a grouped split if ID exists.
    gkf = GroupKFold(n_splits=5)
    train_idx, test_idx = next(gkf.split(X_df, y, groups=ids))
    
    X_train_df, X_test_df = X_df.iloc[train_idx], X_df.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train_df)
    X_test_np = scaler.transform(X_test_df)
    
    print("\nTraining LightGBM on BenSParX...")
    lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    lgb_model.fit(X_train_np, y_train)
    probs_lgb = lgb_model.predict_proba(X_test_np)
    preds_lgb = lgb_model.predict(X_test_np)
    acc_lgb = accuracy_score(y_test, preds_lgb)
    print(f"LightGBM Accuracy: {acc_lgb:.4f}")
    
    print("\nTraining KAN on BenSParX...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_t = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)
    
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
    
    print("\n--- Fusion (Soft Voting) on BenSParX ---")
    fused_probs = (probs_lgb + probs_kan_t) / 2
    fused_preds = np.argmax(fused_probs, axis=1)
    
    acc_fusion = accuracy_score(y_test, fused_preds)
    print(f"Fusion Accuracy: {acc_fusion:.4f}")

if __name__ == '__main__':
    main()
