import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import numpy as np

class ColdStartVisualizer:
    """
    Generates high-fidelity visualizations for Data Scarcity 
    and Saturation research.
    """
    
    def __init__(self, style="seaborn-v0_8-whitegrid"):
        plt.style.use(style)

    def plot_crossover(self, df_a, df_b, label_a="Baseline", label_b="Golden Model"):
        """
        Visualizes the 'Complexity Tax' and the point where 
        Model B justifies its existence.
        """
        plt.figure(figsize=(10, 6))
        
        plt.plot(df_a['n_samples'], df_a['mae'], 'o--', label=label_a, color='#808080', alpha=0.6)
        plt.plot(df_b['n_samples'], df_b['mae'], 'o-', label=label_b, color='#2E86C1', linewidth=2)
        
        plt.xscale('log') # Essential for seeing the small-data 'struggle'
        plt.title(f"The Crossover Audit: {label_a} vs. {label_b}", fontsize=14)
        plt.xlabel("Training Samples (n) - Log Scale", fontsize=12)
        plt.ylabel("Mean Absolute Error ($)", fontsize=12)
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        
        return plt

    def plot_saturation_trajectory(self, n_vals, mae_vals, popt, power_law_func):
        """
        Plots the actual data points against the fitted Power Law curve.
        Visualizes the 'Irreducible Error' (b).
        """
        a, alpha, b = popt
        n_future = np.geomspace(min(n_vals), max(n_vals) * 2, 100)
        mae_pred = power_law_func(n_future, a, alpha, b)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(n_vals, mae_vals, color='red', label='Actual Observations')
        plt.plot(n_future, mae_pred, 'k--', label=f'Power Law Fit (α={alpha:.2f})')
        
        # Draw the 'Asymptotic Floor' (b)
        plt.axhline(y=b, color='green', linestyle=':', label=f'Irreducible Error (${b:,.0f})')
        
        plt.title("The Saturation Ceiling: Predicting Diminishing Returns", fontsize=14)
        plt.xlabel("Training Samples (n)", fontsize=12)
        plt.ylabel("MAE ($)", fontsize=12)
        plt.legend()
        
        return plt

if __name__ == "__main__":
    # ---------------------------------------------------------
    # Example Usage: Generate Demo Plots
    # ---------------------------------------------------------

    print("Running Visualization Demo...")

    # 1. Create Synthetic Data
    n_samples = [100, 500, 1000, 5000, 10000]
    
    # Model A: Baseline (High Bias)
    mae_a = [55000, 52000, 51000, 50500, 50200]
    df_a = pd.DataFrame({'n_samples': n_samples, 'mae': mae_a})

    # Model B: Golden Model (High Variance initially)
    mae_b = [70000, 48000, 42000, 35000, 31000]
    df_b = pd.DataFrame({'n_samples': n_samples, 'mae': mae_b})

    # 2. Instantiate Visualizer
    # Note: seaborn styles might vary by version, fallback to 'ggplot' if needed
    try:
        viz = ColdStartVisualizer()
    except:
        viz = ColdStartVisualizer(style="ggplot")

    # 3. Plot Crossover
    print("Generating Crossover Plot...")
    plt = viz.plot_crossover(df_a, df_b, label_a="Linear Regression", label_b="Random Forest")
    plt.savefig('demo_crossover_plot.png')
    print("Saved 'demo_crossover_plot.png'")
    # plt.show() # blocking

    # 4. Plot Saturation Trajectory
    # Need fit parameters first
    from scipy.optimize import curve_fit
    
    def power_law(n, a, alpha, b):
        return a * np.power(n, -alpha) + b

    # Fit Model B
    popt, _ = curve_fit(power_law, n_samples, mae_b, p0=[100000, 0.5, 20000], maxfev=10000)

    print("Generating Saturation Plot...")
    plt = viz.plot_saturation_trajectory(n_samples, mae_b, popt, power_law)
    plt.savefig('demo_saturation_plot.png')
    print("Saved 'demo_saturation_plot.png'")
    # plt.show() # blocking