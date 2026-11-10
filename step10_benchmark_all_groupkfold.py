import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
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

def extract_pred(preds):
    cols = [c for c in preds.columns if 'prediction' in c.lower()]
    if cols:
        return preds[cols[0]].values
    return preds.iloc[:, -1].values

def main():
    print("Loading Turkish Dataset...")
    df = pd.read_csv('pd_speech_features.csv', header=1)
    y = df['class'].values
    groups = df['id'].values
    X_df = df.drop(columns=['class', 'id'])
    num_cols = list(X_df.columns)

    gkf = GroupKFold(n_splits=5)
    
    models = {
        'LightGBM': [],
        'XGBoost': [],
        'CatBoost': [],
        'KAN': [],
        'TabNet': [],
        'NODE': [],
        'TabTransformer': [],
        'GATE': [],
        'SAINT': [],
        'Fusion (LGBM+KAN)': []
    }

    try:
        from pytorch_tabular import TabularModel
        from pytorch_tabular.models import NodeConfig, TabTransformerConfig, GatedAdditiveTreeEnsembleConfig, FTTransformerConfig
        from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
        tabular_available = True
        
        data_config = DataConfig(target=['target'], continuous_cols=num_cols, categorical_cols=[])
        trainer_config = TrainerConfig(batch_size=128, max_epochs=10, early_stopping="valid_loss", early_stopping_patience=3, checkpoints=None, progress_bar="none")
        optimizer_config = OptimizerConfig()
    except:
        tabular_available = False
        print("pytorch_tabular not available!")

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_df, y, groups=groups)):
        print(f"\n--- Fold {fold+1}/5 ---")
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
        models['LightGBM'].append(accuracy_score(y_test, preds_lgb))
        
        # 2. XGBoost
        xgb_model = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
        xgb_model.fit(X_train_s, y_train)
        preds_xgb = xgb_model.predict(X_test_s)
        models['XGBoost'].append(accuracy_score(y_test, preds_xgb))
        
        # 3. CatBoost
        cat_model = CatBoostClassifier(random_state=42, verbose=0, iterations=100)
        cat_model.fit(X_train_s, y_train)
        preds_cat = cat_model.predict(X_test_s)
        models['CatBoost'].append(accuracy_score(y_test, preds_cat))
        
        # 4. KAN
        X_train_t = torch.tensor(X_train_s, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
        X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)
        probs_kan = train_kan_fold(X_train_t, y_train_t, X_test_t, X_train_s.shape[1], seed=42+fold)
        preds_kan = np.argmax(probs_kan, axis=1)
        models['KAN'].append(accuracy_score(y_test, preds_kan))
        
        # 10. Fusion (LGBM+KAN Soft Voting)
        probs_fused = (probs_lgb + probs_kan) / 2.0
        preds_fused = np.argmax(probs_fused, axis=1)
        models['Fusion (LGBM+KAN)'].append(accuracy_score(y_test, preds_fused))

        # 5. TabNet
        tabnet = TabNetClassifier(verbose=0)
        tabnet.fit(X_train=X_train_s, y_train=y_train, eval_set=[(X_test_s, y_test)], patience=3, max_epochs=10)
        preds_tabnet = tabnet.predict(X_test_s)
        models['TabNet'].append(accuracy_score(y_test, preds_tabnet))
        
        if tabular_available:
            train_data = pd.DataFrame(X_train_s, columns=num_cols)
            train_data['target'] = y_train
            test_data = pd.DataFrame(X_test_s, columns=num_cols)
            test_data['target'] = y_test
            
            # 6. NODE
            try:
                node = TabularModel(data_config=data_config, model_config=NodeConfig(task="classification", num_layers=1, num_trees=64, depth=4), optimizer_config=optimizer_config, trainer_config=trainer_config, verbose=False)
                node.fit(train=train_data)
                models['NODE'].append(accuracy_score(y_test, extract_pred(node.predict(test_data))))
            except: pass
            
            # 7. TabTransformer
            try:
                tt = TabularModel(data_config=data_config, model_config=TabTransformerConfig(task="classification", num_attn_blocks=2, num_heads=2), optimizer_config=optimizer_config, trainer_config=trainer_config, verbose=False)
                tt.fit(train=train_data)
                models['TabTransformer'].append(accuracy_score(y_test, extract_pred(tt.predict(test_data))))
            except: pass
            
            # 8. GATE
            try:
                gate = TabularModel(data_config=data_config, model_config=GatedAdditiveTreeEnsembleConfig(task="classification", gflu_stages=2, num_trees=10, tree_depth=4), optimizer_config=optimizer_config, trainer_config=trainer_config, verbose=False)
                gate.fit(train=train_data)
                models['GATE'].append(accuracy_score(y_test, extract_pred(gate.predict(test_data))))
            except: pass
            
            # 9. SAINT
            try:
                saint = TabularModel(data_config=data_config, model_config=FTTransformerConfig(task="classification", num_attn_blocks=2, num_heads=2), optimizer_config=optimizer_config, trainer_config=trainer_config, verbose=False)
                saint.fit(train=train_data)
                models['SAINT'].append(accuracy_score(y_test, extract_pred(saint.predict(test_data))))
            except: pass

    print("\n=======================================================")
    print("Final Results (Patient-wise 5-Fold GroupKFold on Turkish Dataset)")
    print("=======================================================")
    # Sort models by mean accuracy
    results = []
    for k, v in models.items():
        if len(v) > 0:
            results.append((k, np.mean(v)*100, np.std(v)*100))
    
    results.sort(key=lambda x: x[1], reverse=True)
    for k, mean_acc, std_acc in results:
        print(f"{k:20s}: {mean_acc:.2f}% ± {std_acc:.2f}%")

if __name__ == '__main__':
    main()
