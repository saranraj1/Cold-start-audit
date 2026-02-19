# Cold Start Audit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight library to audit the "Data Hunger" of Machine Learning architectures. It quantifies how much data is needed to outperform a simple baseline and predicts the saturation point of model performance.

## 🚀 Features

*   **Scarcity Simulation**: Systematically train models on subsets of data to measure performance degradation.
*   **Crossover Detection**: Pinpoint the exact sample size (n) where complex models start to justify their complexity.
*   **Saturation Analysis**: Fit Power Law learning curves to predict future performance and irreducible error.
*   **Automated Visualization**: Generate professional plots for audit reports.

## 📂 Project Structure

```
cold-start-audit/
├── data/
│   ├── raw/                # Original datasets (e.g., California Housing)
│   └── samples/            # Subsets for quick testing
├── models/
│   └── checkpoints/        # Saved model artifacts from simulations
├── notebooks/
│   ├── scarcity_simulation.ipynb  # Core simulation logic
│   └── saturation_analysis.ipynb  # Crossover & Power Law analysis
├── reports/
│   └── audit.md           # Generated audit report
├── src/
│   ├── __init__.py
│   ├── cold_start_engine.py  # Main simulation engine
│   ├── saturation_utils.py   # Math utilities for curve fitting
│   └── viz_utils.py          # Plotting utilities
├── requirements.txt
└── README.md
```

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/cold-start-audit.git
cd cold-start-audit
pip install -r requirements.txt
```

## Usage

### 1. Run the Scarcity Simulation
Execute the simulation to gather performance metrics across data milestones.

```bash
python notebooks/scarcity_simulation.py
```
*Note: This script will save a plot to `notebooks/scarcity_plot.png`.*

### 2. Analyze Saturation & Crossover
Use the analysis notebook or utility script to find the crossover point.

```bash
python src/saturation_utils.py
```

### 3. Generate Visualizations
Create custom plots using the visualization utility.

```bash
python src/viz_utils.py
```

## 📊 Example Findings (California Housing Dataset)

*   **Baseline**: Linear Regression
*   **Golden Model**: Random Forest Regressor
*   **Crossover Point**: n = 250
*   **Conclusion**: For datasets smaller than 250 samples, a simple Linear Regression model outperformed the Random Forest.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
