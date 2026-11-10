import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from kan import KAN

def main():
    print("Loading data...")
    df = pd.read_csv('pd_speech_features.csv', header=1)
    
    y = df['class'].values
    X_df = df.drop(columns=['class'])
    
    X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train_df)
    X_test_np = scaler.transform(X_test_df)
    
    # KAN models can be slow with 754 features. Let's do feature selection or use a smaller hidden layer.
    # We will try passing all 754 features first.
    input_dim = X_train_np.shape[1]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_train_t = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)
    
    print("\nTraining Kolmogorov-Arnold Network (KAN)...")
    try:
        # Define KAN model. [input_dim, hidden_dim, output_classes]
        model = KAN(width=[input_dim, 10, 2], grid=5, k=3, seed=42, device=device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        epochs = 100
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(X_train_t)
            loss = criterion(output, y_train_t)
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
                
        model.eval()
        with torch.no_grad():
            output = model(X_test_t)
            preds = torch.argmax(output, dim=1).cpu().numpy()
            acc = accuracy_score(y_test, preds)
            
        print(f"\nKolmogorov-Arnold Network (KAN) Accuracy: {acc:.4f}")

    except Exception as e:
        print(f"KAN failed: {e}")

if __name__ == '__main__':
    main()
