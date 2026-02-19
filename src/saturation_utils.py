import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def power_law(n, a, alpha, b):
    """
    Standard Learning Curve Power Law: E(n) = a * n^-alpha + b
    alpha: The 'Learning Rate' (how quickly the model absorbs new info)
    b: The irreducible error (asymptotic performance)
    """
    return a * np.power(n, -alpha) + b

class SaturationAnalyst:
    """
    Mathematical tools to find the 'Diminishing Returns' threshold.
    """
    
    @staticmethod
    def calculate_crossover(results_a, results_b):
        """
        Finds the sample size (n) where Model B starts outperforming Model A.
        """
        merged = pd.merge(results_a, results_b, on='n_samples', suffixes=('_A', '_B'))
        # Find where the difference in MAE changes sign
        merged['diff'] = merged['mae_A'] - merged['mae_B']
        
        # Identification of the Crossover Point
        crossover = merged[merged['diff'] > 0].iloc[0] if any(merged['diff'] > 0) else None
        return crossover['n_samples'] if crossover is not None else "No Crossover Detected"

    @staticmethod
    def fit_learning_trajectory(n_vals, mae_vals):
        """
        Fits the data to the Power Law to predict future performance.
        """
        popt, _ = curve_fit(power_law, n_vals, mae_vals, p0=[100000, 0.5, 20000], maxfev=10000)
        return popt # Returns [a, alpha, b]

if __name__ == "__main__":
    # ---------------------------------------------------------
    # Example Usage: Analyze Synthetic Learning Curves
    # ---------------------------------------------------------

    print("Running Saturation Analysis Demo...")

    # 1. Create Synthetic Data (Simulating a specialized vs general model comparison)
    # Model A: Simple model (Fast start, high plateau)
    n_samples = [100, 500, 1000, 5000, 10000]
    
    # Simulating MAE values: y = a * x^-alpha + b
    # A: a=50000, alpha=0.3, b=20000
    mae_a = [50000 * (n**-0.3) + 20000 for n in n_samples]
    
    # Model B: Complex model (Slow start, low plateau)
    # B: a=100000, alpha=0.5, b=10000
    mae_b = [100000 * (n**-0.5) + 10000 for n in n_samples]

    df_a = pd.DataFrame({'n_samples': n_samples, 'mae': mae_a})
    df_b = pd.DataFrame({'n_samples': n_samples, 'mae': mae_b})

    print("\nModel A Data (Simple):")
    print(df_a)
    print("\nModel B Data (Complex):")
    print(df_b)

    # 2. Fit Learning Curve for Model B
    print("\nFitting Learning Curve for Model B...")
    try:
        params = SaturationAnalyst.fit_learning_trajectory(df_b['n_samples'], df_b['mae'])
        print(f"Fitted Parameters (a, alpha, b): {params}")
        print(f"Predicted Error at 50,000 samples: {power_law(50000, *params):.2f}")
    except Exception as e:
        print(f"Fitting failed: {e}")

    # 3. Find Crossover Point
    # Note: calculate_crossover expects dataframes with 'n_samples' and 'mae' cols, 
    # but the logic inside creates suffixes _A and _B.
    # The helper function merges them.
    # Let's adjust the column names to match what the function expects or renaming isn't needed if 
    # the function does the suffixing automatically on collision.
    # The function expects 'mae' column in both.
    
    print("\nCalculating Crossover Point...")
    crossover = SaturationAnalyst.calculate_crossover(df_a, df_b)
    print(f"Crossover Sample Size: {crossover}")