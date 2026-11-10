import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from pytorch_tabnet.tab_model import TabNetClassifier
import lightgbm as lgb
import torch
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Loading data for Fusion...")
    df = pd.read_csv('pd_speech_features.csv', header=1)
    
    y = df['class'].values
    X = df.drop(columns=['class']).values
    
    # Standardize features (important for TabNet)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train_tab, X_valid, y_train_tab, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    print("Training Best Traditional Boosting Model (LightGBM)...")
    lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    lgb_model.fit(X_train, y_train)
    
    # Note: Replace these parameters with the best ones found in step 2. 
    # For now, we'll use robust defaults for TabNet if step 2 results aren't explicitly passed.
    print("Training Best Lightweight Transformer (TabNet)...")
    tabnet_model = TabNetClassifier(
        n_d=16, n_a=16, n_steps=3, gamma=1.0,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size":10, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax',
        verbose=0
    )
    
    tabnet_model.fit(
        X_train=X_train_tab, y_train=y_train_tab,
        eval_set=[(X_valid, y_valid)],
        eval_name=['valid'],
        eval_metric=['accuracy'],
        max_epochs=100, patience=15,
        batch_size=64, virtual_batch_size=32,
        num_workers=0, drop_last=False
    )
    
    print("\n--- Testing Individual Models ---")
    preds_lgb = lgb_model.predict(X_test)
    acc_lgb = accuracy_score(y_test, preds_lgb)
    print(f"LightGBM Accuracy: {acc_lgb:.4f}")
    
    preds_tab = tabnet_model.predict(X_test)
    acc_tab = accuracy_score(y_test, preds_tab)
    print(f"TabNet Accuracy: {acc_tab:.4f}")
    
    print("\n--- Fusion (Soft Voting) ---")
    # Get probabilities
    probs_lgb = lgb_model.predict_proba(X_test)
    probs_tab = tabnet_model.predict_proba(X_test)
    
    # Average probabilities
    fused_probs = (probs_lgb + probs_tab) / 2
    
    # Get final predictions
    fused_preds = np.argmax(fused_probs, axis=1)
    
    acc_fusion = accuracy_score(y_test, fused_preds)
    print(f"Fusion (LightGBM + TabNet) Accuracy: {acc_fusion:.4f}")
    
    print("\nClassification Report (Fusion):")
    print(classification_report(y_test, fused_preds))

if __name__ == '__main__':
    main()
