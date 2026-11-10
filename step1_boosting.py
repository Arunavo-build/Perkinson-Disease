import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

def evaluate_model(model, X, y, groups=None, cv_type='random'):
    if cv_type == 'random':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if isinstance(model, CatBoostClassifier):
            model.fit(X_train, y_train, verbose=False)
        elif isinstance(model, xgb.XGBClassifier):
            model.fit(X_train, y_train, verbose=False)
        else:
            model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return accuracy_score(y_test, preds)
    elif cv_type == 'group':
        gkf = GroupKFold(n_splits=5)
        scores = []
        for train_idx, test_idx in gkf.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            if isinstance(model, CatBoostClassifier):
                model.fit(X_train, y_train, verbose=False)
            elif isinstance(model, xgb.XGBClassifier):
                model.fit(X_train, y_train, verbose=False)
            else:
                model.fit(X_train, y_train)
            preds = model.predict(X_test)
            scores.append(accuracy_score(y_test, preds))
        return np.mean(scores)

def main():
    print("Loading data...")
    df = pd.read_csv('pd_speech_features.csv', header=1)
    
    y = df['class']
    groups = df['id']
    
    # 1. With patient (including 'id' as feature)
    X_with_id = df.drop(columns=['class'])
    
    # 2. Without patient (excluding 'id' as feature)
    X_without_id = df.drop(columns=['id', 'class'])

    models = {
        'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=False)
    }

    results = []

    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        # Interpretation 1: 'With patient' means keeping 'id' column as feature, 'Without patient' means dropping it. Both random split.
        acc_with_id = evaluate_model(model, X_with_id, y, cv_type='random')
        acc_without_id = evaluate_model(model, X_without_id, y, cv_type='random')
        
        # Interpretation 2: 'With patient' means Random Split (patient records in both train and test), 
        # 'Without patient' means GroupKFold (patient completely isolated in test).
        # We will test GroupKFold without ID to see the true patient-independent performance.
        acc_group_without_id = evaluate_model(model, X_without_id, y, groups=groups, cv_type='group')

        results.append({
            'Model': name,
            'With ID (Random Split)': round(acc_with_id, 4),
            'Without ID (Random Split)': round(acc_without_id, 4),
            'Without ID (Patient Grouped)': round(acc_group_without_id, 4)
        })

    results_df = pd.DataFrame(results)
    print("\nResults:")
    print(results_df.to_string(index=False))

if __name__ == '__main__':
    main()
