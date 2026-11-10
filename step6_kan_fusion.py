import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb
from kan import KAN
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Loading data for LightGBM + KAN Fusion...")
    df = pd.read_csv('pd_speech_features.csv', header=1)
    
    y = df['class'].values
    X_df = df.drop(columns=['class'])
    
    # Random split (With Patient ID) - as this gave the best baseline performance
    X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train_df)
    X_test_np = scaler.transform(X_test_df)
    
    print("\nTraining LightGBM...")
    lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    lgb_model.fit(X_train_np, y_train)
    probs_lgb = lgb_model.predict_proba(X_test_np)
    preds_lgb = lgb_model.predict(X_test_np)
    acc_lgb = accuracy_score(y_test, preds_lgb)
    print(f"LightGBM Accuracy: {acc_lgb:.4f}")
    
    print("\nTraining KAN (Kolmogorov-Arnold Network)...")
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
        # Convert logits to probabilities using Softmax
        probs_kan_t = torch.softmax(logits_kan, dim=1).cpu().numpy()
        preds_kan = np.argmax(probs_kan_t, axis=1)
    acc_kan = accuracy_score(y_test, preds_kan)
    print(f"KAN Accuracy: {acc_kan:.4f}")
    
    print("\n--- Fusion (Soft Voting: LightGBM + KAN) ---")
    # Soft Voting by averaging probabilities
    fused_probs = (probs_lgb + probs_kan_t) / 2
    fused_preds = np.argmax(fused_probs, axis=1)
    
    acc_fusion = accuracy_score(y_test, fused_preds)
    print(f"Fusion Accuracy: {acc_fusion:.4f}")
    
    print("\nClassification Report (Fusion):")
    print(classification_report(y_test, fused_preds))

if __name__ == '__main__':
    main()
