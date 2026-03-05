import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

def train_and_save():
    # 1. Setup paths
    base_path = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_path, 'processed_cleveland.csv')
    models_dir = os.path.join(base_path, 'models')
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # 2. Load and Clean
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    # Handle the '?' values found in your specific dataset
    df.replace('?', np.nan, inplace=True)
    df = df.dropna()
    
    # Convert objects to numeric
    df['ca'] = pd.to_numeric(df['ca'])
    df['thal'] = pd.to_numeric(df['thal'])

    # 3. Features and Target
    # Convert 'num' (0-4) to binary target: 0 = Healthy, 1 = Disease
    y = df['num'].apply(lambda x: 1 if x > 0 else 0)
    X = df.drop(['num'], axis=1)

    # 4. Train
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 5. Save
    model_path = os.path.join(models_dir, 'heart_disease_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✅ Success! Model saved in: {model_path}")

if __name__ == "__main__":
    train_and_save()
