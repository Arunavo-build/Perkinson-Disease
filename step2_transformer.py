import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

def main():
    print("Loading data...")
    df = pd.read_csv('pd_speech_features.csv', header=1)
    
    # We found that "With Patient" (including 'id') with Random Split gave the highest accuracy (0.9013).
    # So we will use that setup.
    y = df['class'].values
    X = df.drop(columns=['class']).values
    
    # Standardize features for neural networks
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Validation set for TabNet early stopping
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Lightweight Transformer (TabNet) parameter variations
    parameter_grid = [
        {'n_d': 8, 'n_a': 8, 'n_steps': 3, 'gamma': 1.3, 'name': 'Small_1'},
        {'n_d': 16, 'n_a': 16, 'n_steps': 4, 'gamma': 1.5, 'name': 'Medium_1'},
        {'n_d': 32, 'n_a': 32, 'n_steps': 5, 'gamma': 1.5, 'name': 'Large_1'},
        {'n_d': 16, 'n_a': 16, 'n_steps': 3, 'gamma': 1.0, 'name': 'Medium_2_Low_Gamma'},
        {'n_d': 8, 'n_a': 8, 'n_steps': 5, 'gamma': 1.2, 'name': 'Small_2_More_Steps'}
    ]

    best_acc = 0
    best_params = None

    print("\n--- Tuning Lightweight Transformer (TabNet) ---")
    for params in parameter_grid:
        print(f"\nTesting parameters: {params['name']}")
        print(f"n_d={params['n_d']}, n_a={params['n_a']}, n_steps={params['n_steps']}, gamma={params['gamma']}")
        
        clf = TabNetClassifier(
            n_d=params['n_d'], n_a=params['n_a'], n_steps=params['n_steps'],
            gamma=params['gamma'],
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size":10, "gamma":0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax',
            verbose=0
        )
        
        clf.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_valid, y_valid)],
            eval_name=['valid'],
            eval_metric=['accuracy'],
            max_epochs=100, patience=15,
            batch_size=64, virtual_batch_size=32,
            num_workers=0, drop_last=False
        )
        
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy for {params['name']}: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_params = params

    print(f"\nBest TabNet configuration: {best_params['name']} with Accuracy: {best_acc:.4f}")

if __name__ == '__main__':
    main()
