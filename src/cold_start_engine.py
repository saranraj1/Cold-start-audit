import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os

class ColdStartEngine:
    """
    Research Engine to quantify the 'Data Hunger' of ML architectures.
    Simulates the evolution of a model from Scarcity to Saturation.
    """
    
    def __init__(self, model_factory, project_root="."):
        self.model_factory = model_factory
        self.results = []
        self.checkpoint_dir = os.path.join(project_root, "models/checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def run_scarcity_audit(self, X_train_full, y_train_full, X_test, y_test, milestones):
        """
        Runs the full audit across defined data volume milestones.
        
        Logic: At each milestone, we train a fresh model and measure its 
        generalization gap on a fixed, unseen test set.
        """
        print(f"Starting Scarcity Audit across {len(milestones)} milestones...")
        
        for n in milestones:
            # 1. Stratified-style sampling (using random_state for reproducibility)
            # We take exactly 'n' samples from the training universe.
            if n > len(X_train_full):
                continue
                
            if n == len(X_train_full):
                X_slice, y_slice = X_train_full, y_train_full
            else:
                X_slice, _, y_slice, _ = train_test_split(
                    X_train_full, y_train_full, train_size=n, random_state=42
                )
            
            # 2. Re-initialize and Train
            # We use a factory to ensure no 'memory' carries over between milestones.
            model = self.model_factory()
            model.fit(X_slice, y_slice)
            
            # 3. Evaluate Performance
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            
            # 4. Record Metadata
            # Storing MAE and the ratio of MAE to the data volume.
            self.results.append({
                'n_samples': n,
                'mae': mae,
                'efficiency_score': mae * n  # Lower is better (Cost of Accuracy)
            })
            
            # 5. Persist Checkpoint
            # Saving the 'Small-Data Version' for future stress tests.
            joblib.dump(model, os.path.join(self.checkpoint_dir, f"model_n_{n}.pkl"))
            
            print(f"   [n={n:5}] MAE: ${mae:,.2f}")
            
        return pd.DataFrame(self.results)

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor
    
    # ---------------------------------------------------------
    # Example Usage: Run Scarcity Audit on Housing Data
    # ---------------------------------------------------------
    
    # 1. Load Data
    # handle execution from root or src
    if os.path.exists(os.path.join("data", "raw", "housing.csv")):
        data_path = os.path.join("data", "raw", "housing.csv")
    elif os.path.exists(os.path.join("..", "data", "raw", "housing.csv")):
        data_path = os.path.join("..", "data", "raw", "housing.csv")
    else:
        raise FileNotFoundError("Could not find housing.csv in data/raw or ../data/raw")
        
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Simple preprocessing
    df = df.dropna()
    df = pd.get_dummies(df, drop_first=True)
    
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Define Model Factory
    def rf_model_factory():
        return RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        
    # 4. Initialize Engine
    # If run from src, project_root might need adjustment if using default "."
    # But since we are passing model_factory and using absolute path for data loading outside, 
    # the internal checkpoint_dir usage depends on project_root.
    # Default is ".". If running from src, "." is src. "models/checkpoints" will be created in src/models/checkpoints.
    # We might want to set project_root to ".." if running from src.
    if os.path.basename(os.getcwd()) == "src":
        project_root = ".."
    else:
        project_root = "."
        
    engine = ColdStartEngine(model_factory=rf_model_factory, project_root=project_root)
    
    # 5. Define Milestones
    # valid milestones
    milestones = [100, 500, 1000, 5000, len(X_train)]
    
    # 6. Run Audit
    audit_results = engine.run_scarcity_audit(X_train, y_train, X_test, y_test, milestones)
    
    print("\nAudit Results:")
    print(audit_results)