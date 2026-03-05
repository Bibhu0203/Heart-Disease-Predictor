import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Get the directory where THIS script (train.py) is saved
base_path = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the script location
csv_path = os.path.join(base_path, 'processed_cleveland.csv')
models_dir = os.path.join(base_path, 'models')
model_output_path = os.path.join(models_dir, 'heart_disease_model.pkl')

print(f"--- Diagnostic ---")
print(f"Script location: {base_path}")
print(f"Looking for CSV at: {csv_path}")

# 1. Check if CSV exists
if not os.path.exists(csv_path):
    print(f"❌ ERROR: Cannot find 'processed_cleveland.csv' at {csv_path}")
    print("Please make sure the CSV file is in the same folder as this train.py script.")
else:
    try:
        # 2. Create models directory
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            print("Created 'models' folder.")

        # 3. Load and Clean
        df = pd.read_csv(csv_path)
        df.replace('?', np.nan, inplace=True)
        df = df.dropna()
        df['ca'] = pd.to_numeric(df['ca'])
        df['thal'] = pd.to_numeric(df['thal'])
        
        # 0 = healthy, 1-4 = sick -> convert to 0 and 1
        df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
        X = df.drop(['num', 'target'], axis=1)
        y = df['target']

        # 4. Train
        print("Training model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        # 5. Save
        with open(model_output_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"✅ SUCCESS! Model saved at: {model_output_path}")

    except Exception as e:
        print(f"❌ AN ERROR OCCURRED: {e}")