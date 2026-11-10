import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Loading data...")
    df = pd.read_csv('pd_speech_features.csv', header=1)
    
    y = df['class'].values
    X_df = df.drop(columns=['class'])
    
    X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train_df)
    X_test_np = scaler.transform(X_test_df)
    
    results = {}
    
    print("\nTraining TabPFN...")
    try:
        from tabpfn import TabPFNClassifier
        from sklearn.feature_selection import SelectKBest, f_classif
        skb = SelectKBest(f_classif, k=100)
        X_train_pfn = skb.fit_transform(X_train_np, y_train)
        X_test_pfn = skb.transform(X_test_np)
        
        tabpfn_model = TabPFNClassifier(device='cpu') # Removed bad kwargs
        tabpfn_model.fit(X_train_pfn, y_train)
        preds_pfn = tabpfn_model.predict(X_test_pfn)
        acc_pfn = accuracy_score(y_test, preds_pfn)
        print(f"TabPFN Accuracy: {acc_pfn:.4f}")
        results['TabPFN'] = acc_pfn
    except Exception as e:
        print(f"TabPFN failed: {e}")

    print("\nPreparing PyTorch Tabular...")
    try:
        from pytorch_tabular import TabularModel
        from pytorch_tabular.models import (
            NodeConfig,
            TabTransformerConfig,
            GatedAdditiveTreeEnsembleConfig,
            FTTransformerConfig
        )
        from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig

        train_data = X_train_df.copy()
        train_data['target'] = y_train
        test_data = X_test_df.copy()
        test_data['target'] = y_test
        
        num_cols = list(X_train_df.columns)
        
        data_config = DataConfig(
            target=['target'],
            continuous_cols=num_cols,
            categorical_cols=[],
        )
        trainer_config = TrainerConfig(
            batch_size=128,
            max_epochs=10,
            early_stopping="valid_loss",
            early_stopping_patience=3,
            checkpoints=None,
            progress_bar="none"
        )
        optimizer_config = OptimizerConfig()
        
        def extract_pred(preds):
            cols = [c for c in preds.columns if 'prediction' in c.lower()]
            if cols:
                return preds[cols[0]].values
            return preds.iloc[:, -1].values

        print("\nTraining NODE...")
        try:
            node_config = NodeConfig(
                task="classification",
                num_layers=1,
                num_trees=64,
                depth=4
            )
            node_model = TabularModel(
                data_config=data_config, model_config=node_config, 
                optimizer_config=optimizer_config, trainer_config=trainer_config,
                verbose=False
            )
            node_model.fit(train=train_data)
            preds_node = extract_pred(node_model.predict(test_data))
            acc_node = accuracy_score(y_test, preds_node)
            print(f"NODE Accuracy: {acc_node:.4f}")
            results['NODE'] = acc_node
        except Exception as e:
            print(f"NODE failed: {e}")

        print("\nTraining TabTransformer...")
        try:
            tab_tf_config = TabTransformerConfig(
                task="classification",
                num_attn_blocks=2,
                num_heads=2,
            )
            tab_tf_model = TabularModel(
                data_config=data_config, model_config=tab_tf_config, 
                optimizer_config=optimizer_config, trainer_config=trainer_config,
                verbose=False
            )
            tab_tf_model.fit(train=train_data)
            preds_tt = extract_pred(tab_tf_model.predict(test_data))
            acc_tt = accuracy_score(y_test, preds_tt)
            print(f"TabTransformer Accuracy: {acc_tt:.4f}")
            results['TabTransformer'] = acc_tt
        except Exception as e:
            print(f"TabTransformer failed: {e}")
            
        print("\nTraining GATE (Gated Additive Tree Ensemble)...")
        try:
            gate_config = GatedAdditiveTreeEnsembleConfig(
                task="classification",
                gflu_stages=2,
                num_trees=10,
                tree_depth=4
            )
            gate_model = TabularModel(
                data_config=data_config, model_config=gate_config, 
                optimizer_config=optimizer_config, trainer_config=trainer_config,
                verbose=False
            )
            gate_model.fit(train=train_data)
            preds_gate = extract_pred(gate_model.predict(test_data))
            acc_gate = accuracy_score(y_test, preds_gate)
            print(f"GatedTabTransformer/GATE Accuracy: {acc_gate:.4f}")
            results['GatedTabTransformer'] = acc_gate
        except Exception as e:
            print(f"GATE failed: {e}")
            
        print("\nTraining FTTransformer (SAINT variant)...")
        try:
            ftt_config = FTTransformerConfig(
                task="classification",
                num_attn_blocks=2,
                num_heads=2,
            )
            ftt_model = TabularModel(
                data_config=data_config, model_config=ftt_config, 
                optimizer_config=optimizer_config, trainer_config=trainer_config,
                verbose=False
            )
            ftt_model.fit(train=train_data)
            preds_ftt = extract_pred(ftt_model.predict(test_data))
            acc_ftt = accuracy_score(y_test, preds_ftt)
            print(f"SAINT/FTTransformer Accuracy: {acc_ftt:.4f}")
            results['SAINT/FTTransformer'] = acc_ftt
        except Exception as e:
            print(f"FTTransformer failed: {e}")

    except Exception as e:
        print(f"PyTorch Tabular failed: {e}")

    print("\n--- Final Results ---")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

if __name__ == '__main__':
    main()
