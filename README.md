Maximum Likelihood Estimation & EM Algorithm – Assignment 3
!Python
!Status
!License
📋 Project Overview
This project implements Maximum Likelihood Estimation (MLE) and the Expectation-Maximization (EM) algorithm to analyze Gaussian data with missing values. It compares different parameter estimation strategies and highlights how missing data affects the accuracy of statistical estimates.
🎯 Goals

✅ Implement MLE for Gaussian distributions (1D, 2D, 3D)
✅ Develop EM algorithm to handle missing data
✅ Compare results between complete and incomplete datasets
✅ Visualize clusters and algorithm convergence
✅ Analyze the impact of different initialization strategies

📊 Datasets
Category ω₁ (Omega 1)

10 three-dimensional points with features [x₁, x₂, x₃]
Missing data: x₃ is missing in even-indexed points (2, 4, 6, 8, 10)
Missing rate: 50% in the x₃ dimension

Category ω₂ (Omega 2)

10 complete three-dimensional points
Used for: Separable model with diagonal covariance matrix

🚀 Key Features
1. Traditional MLE (mle_omega1.py)

Univariate estimation for each feature
Bivariate analysis for feature pairs
Full 3D analysis
Comparison of mean and variance estimates

2. EM Algorithm (em_algorithm.py / em_algorithm_ascii.py)

Full EM implementation for handling missing data
Two initialization strategies:

Zero initialization: x₃ = 0
Mean-based initialization: x₃ = (x₁ + x₂)/2


Comparison with complete data (ground truth)
Detailed convergence analysis

3. Advanced Visualizations

3D cluster plots
Convergence tracking
Side-by-side comparisons
Correlation matrices
Feature-wise error analysis

📁 Project Structure
assignment3/
├── 🔧 Core Code
│   ├── mle_omega1.py              # Traditional MLE (complete data)
│   ├── em_algorithm.py            # EM algorithm (Unicode version)
│   └── em_algorithm_ascii.py      # EM algorithm (ASCII version)
│
├── 📊 Visualization & Analysis
│   ├── cluster_visualization.py   # Comprehensive visualizations
│   ├── results_summary.py         # Summary of results
│   └── show_plots.py              # Plot viewer
│
├── 📈 Results (Graphs)
│   ├── em_convergence_analysis.png
│   ├── em_complete_comparison.png
│   └── cluster_analysis_comprehensive.png
│
├── 📚 Documentation
│   ├── README.md                  # This file
│   ├── complete_vs_missing_comparison.md
│   ├── em_summary_report.md
│   └── requirements.txt           # Dependencies
│
└── 📁 Misc
    ├── em_output.txt              # Execution logs
    └── __pycache__/               # Python cache

🛠️ Installation & Usage
Prerequisites
ShellPython 3.13+pip (Python package manager)Mostrar mais linhas
1. Clone the repository
Shellgit clone https://github.com/your-username/mle-em-algorithm-assignment3.gitcd mle-em-algorithm-assignment3Mostrar mais linhas
2. Install dependencies
Shellpip install -r requirements.txt``Mostrar mais linhas
3. Run the algorithms
Traditional MLE (complete data):
Shellpython mle_omega1.pyMostrar mais linhas
EM Algorithm (with missing data):
Shellpython em_algorithm.py# or ASCII-compatible version:python em_algorithm_ascii.pyMostrar mais linhas
View results:
Shellpython show_plots.pypython results_summary.pyMostrar mais linhas
📊 Key Findings
✅ EM Algorithm Successes

Perfect recovery of observed dimensions (x₁, x₂)
Zero error in μ₁, μ₂, σ₁², σ₂²
Robust convergence in 16 iterations
Initialization-independent results

❌ Identified Limitations

Systematic bias in the missing dimension (x₃)
1.684 units error in μ₃ (185% relative error)
61% underestimation in σ₃²
Cluster compression (54% volume reduction)

🎯 Key Insights

MLE preserves information in observed dimensions
Missing data patterns significantly affect estimates
EM is robust, but introduces predictable bias
Cluster structure is partially recoverable

📈 Generated Visuals
1. Convergence Analysis

Log-likelihood evolution
Initialization strategy comparison
Identical convergence behavior

2. Complete vs Missing Comparison

Side-by-side parameter estimates
Feature-wise error analysis
Visual impact of missing data

3. Comprehensive Cluster Analysis

3D data visualization
2D projections
Correlation matrices
Detailed error breakdown

🔬 Theoretical Foundations
Maximum Likelihood Estimation (MLE)
Plain Textmath não tem suporte total. O realce de sintaxe é baseado em Plain Text.μ̂ = (1/n) × Σᵢ xᵢΣ̂ = (1/n) × Σᵢ (xᵢ - μ̂)(xᵢ - μ̂)ᵀ``Mostrar mais linhas
Expectation-Maximization (EM) Algorithm

E-step: E[X₃|X₁,X₂] = μ₃ + Σ₃₁Σ₁₁⁻¹(X₁₂ - μ₁₂)
M-step: Update parameters using "completed" data
Convergence: Based on log-likelihood improvement

🙏 Acknowledgments

Based on concepts from Machine Learning and Statistics
Algorithms grounded in Maximum Likelihood Estimation theory
Visualizations inspired by Data Science best practices
