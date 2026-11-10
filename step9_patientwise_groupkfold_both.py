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
        logits = kan(X_test_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs

def evaluate_patientwise_gkf(df, id_col, target_col, name, n_splits=5, extract_id=False):
    print(f"\n=======================================================")
    print(f"Dataset: {name} (Patient-wise {n_splits}-Fold GroupKFold)")
    print(f"=======================================================")
    
    y = df[target_col].values
    
    if extract_id:
        groups = df[id_col].astype(str).apply(lambda x: x.split('_')[0]).values
    else:
        groups = df[id_col].values
        
    print(f"Total Unique Patients: {len(np.unique(groups))}")
    
    X_df = df.drop(columns=[target_col, id_col])
    
    gkf = GroupKFold(n_splits=n_splits)
    
    lgb_accs, kan_accs, fusion_accs = [], [], []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_df, y, groups=groups)):
        X_train, X_test = X_df.iloc[train_idx].values, X_df.iloc[test_idx].values
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # 1. LightGBM
        lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1, n_estimators=100)
        lgb_model.fit(X_train_s, y_train)
        probs_lgb = lgb_model.predict_proba(X_test_s)
        preds_lgb = np.argmax(probs_lgb, axis=1)
        acc_lgb = accuracy_score(y_test, preds_lgb)
        lgb_accs.append(acc_lgb)
        
        # 2. KAN
        X_train_t = torch.tensor(X_train_s, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
        X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)
        
        probs_kan = train_kan_fold(X_train_t, y_train_t, X_test_t, X_train_s.shape[1], seed=42+fold)
        preds_kan = np.argmax(probs_kan, axis=1)
        acc_kan = accuracy_score(y_test, preds_kan)
        kan_accs.append(acc_kan)
        
        # 3. Soft Voting Fusion
        probs_fused = (probs_lgb + probs_kan) / 2.0
        preds_fused = np.argmax(probs_fused, axis=1)
        acc_fused = accuracy_score(y_test, preds_fused)
        fusion_accs.append(acc_fused)
        
        print(f"Fold {fold+1}/{n_splits} -> LightGBM: {acc_lgb*100:.2f}%, KAN: {acc_kan*100:.2f}%, Fusion: {acc_fused*100:.2f}%")
        
    print(f"\n--- Mean Results across {n_splits} Folds ---")
    print(f"LightGBM Mean Accuracy : {np.mean(lgb_accs)*100:.2f}% ± {np.std(lgb_accs)*100:.2f}%")
    print(f"KAN Mean Accuracy      : {np.mean(kan_accs)*100:.2f}% ± {np.std(kan_accs)*100:.2f}%")
    print(f"Fusion Mean Accuracy   : {np.mean(fusion_accs)*100:.2f}% ± {np.std(fusion_accs)*100:.2f}%")
    
    return np.mean(lgb_accs), np.mean(kan_accs), np.mean(fusion_accs)

def main():
    # 1. Turkish UCI Dataset
    df_turkish = pd.read_csv('pd_speech_features.csv', header=1)
    evaluate_patientwise_gkf(df_turkish, id_col='id', target_col='class', name="Turkish Dataset (UCI)", n_splits=5, extract_id=False)
    
    # 2. Bengali BenSParX Dataset
    df_bengali = pd.read_csv('BenSParX/BenSParX.csv')
    evaluate_patientwise_gkf(df_bengali, id_col='id', target_col='class', name="Bengali Dataset (BenSParX)", n_splits=5, extract_id=True)

if __name__ == '__main__':
    main()
