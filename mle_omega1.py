"""
Maximum Likelihood Estimation (MLE) Analysis
Problem 1: Programming exercises in Maximum Likelihood Estimation and Bayesian Estimation

This program computes:
1. UNIVARIATE: μ̂ and σ̂² for each feature x₁, x₂, x₃ individually (ω₁)
2. BIVARIATE: μ̂ and Σ̂ for all possible pairs of features (ω₁)
3. TRIVARIATE: μ̂ and Σ̂ for the full three-dimensional data (ω₁)
4. SEPARABLE MODEL: μ̂ and diag(σ₁², σ₂², σ₃²) for ω₂ data

MLE Formulas:
UNIVARIATE Gaussian Distribution:
- μ̂ = (1/n) * Σᵢ₌₁ⁿ xᵢ
- σ̂² = (1/n) * Σᵢ₌₁ⁿ (xᵢ - μ̂)²

BIVARIATE Gaussian Distribution:
- μ̂ = (1/n) * Σᵢ₌₁ⁿ xᵢ  (2D mean vector)
- Σ̂ = (1/n) * Σᵢ₌₁ⁿ (xᵢ - μ̂)(xᵢ - μ̂)ᵀ  (2×2 covariance matrix)

TRIVARIATE Gaussian Distribution:
- μ̂ = (1/n) * Σᵢ₌₁ⁿ xᵢ  (3D mean vector)
- Σ̂ = (1/n) * Σᵢ₌₁ⁿ (xᵢ - μ̂)(xᵢ - μ̂)ᵀ  (3×3 covariance matrix)

SEPARABLE TRIVARIATE Gaussian Distribution:
- μ̂ = (1/n) * Σᵢ₌₁ⁿ xᵢ  (3D mean vector)
- Σ̂ = diag(σ₁², σ₂², σ₃²) where σⱼ² = (1/n) * Σᵢ₌₁ⁿ (xᵢⱼ - μⱼ)²
"""

import numpy as np


def mle_univariate_gaussian(data):
    """
    Compute Maximum Likelihood Estimates for univariate Gaussian distribution.
    
    Parameters:
    -----------
    data : array-like
        1D array of observations
    
    Returns:
    --------
    mu_hat : float
        MLE estimate of mean (μ̂)
    sigma_squared_hat : float
        MLE estimate of variance (σ̂²)
    """
    n = len(data)
    
    # MLE for mean: μ̂ = (1/n) * Σxᵢ
    mu_hat = np.mean(data)
    
    # MLE for variance: σ̂² = (1/n) * Σ(xᵢ - μ̂)²
    # Using ddof=0 for population variance (MLE estimate)
    sigma_squared_hat = np.var(data, ddof=0)
    
    return mu_hat, sigma_squared_hat


def mle_bivariate_gaussian(data):
    """
    Compute Maximum Likelihood Estimates for bivariate Gaussian distribution.
    
    Parameters:
    -----------
    data : array-like
        2D array of shape (n, 2) where n is number of observations
    
    Returns:
    --------
    mu_hat : numpy.ndarray
        MLE estimate of mean vector (μ̂) - 2D vector
    Sigma_hat : numpy.ndarray
        MLE estimate of covariance matrix (Σ̂) - 2×2 matrix
    """
    n = data.shape[0]
    
    # MLE for mean vector: μ̂ = (1/n) * Σᵢ₌₁ⁿ xᵢ
    mu_hat = np.mean(data, axis=0)
    
    # MLE for covariance matrix: Σ̂ = (1/n) * Σᵢ₌₁ⁿ (xᵢ - μ̂)(xᵢ - μ̂)ᵀ
    # Using ddof=0 for population covariance (MLE estimate)
    Sigma_hat = np.cov(data, rowvar=False, ddof=0)
    
    return mu_hat, Sigma_hat


def mle_trivariate_gaussian(data):
    """
    Compute Maximum Likelihood Estimates for trivariate (3D) Gaussian distribution.
    
    Parameters:
    -----------
    data : array-like
        2D array of shape (n, 3) where n is number of observations
    
    Returns:
    --------
    mu_hat : numpy.ndarray
        MLE estimate of mean vector (μ̂) - 3D vector
    Sigma_hat : numpy.ndarray
        MLE estimate of covariance matrix (Σ̂) - 3×3 matrix
    """
    n = data.shape[0]
    
    # MLE for mean vector: μ̂ = (1/n) * Σᵢ₌₁ⁿ xᵢ
    mu_hat = np.mean(data, axis=0)
    
    # MLE for covariance matrix: Σ̂ = (1/n) * Σᵢ₌₁ⁿ (xᵢ - μ̂)(xᵢ - μ̂)ᵀ
    # Using ddof=0 for population covariance (MLE estimate)
    Sigma_hat = np.cov(data, rowvar=False, ddof=0)
    
    return mu_hat, Sigma_hat


def mle_separable_trivariate_gaussian(data):
    """
    Compute Maximum Likelihood Estimates for separable trivariate (3D) Gaussian distribution.
    Assumes Σ = diag(σ₁², σ₂², σ₃²) - diagonal covariance matrix (independent features).
    
    Parameters:
    -----------
    data : array-like
        2D array of shape (n, 3) where n is number of observations
    
    Returns:
    --------
    mu_hat : numpy.ndarray
        MLE estimate of mean vector (μ̂) - 3D vector
    sigma_squared_hat : numpy.ndarray
        MLE estimate of diagonal variances [σ₁², σ₂², σ₃²]
    Sigma_diagonal : numpy.ndarray
        Diagonal covariance matrix with off-diagonal elements = 0
    """
    n = data.shape[0]
    
    # MLE for mean vector: μ̂ = (1/n) * Σᵢ₌₁ⁿ xᵢ
    mu_hat = np.mean(data, axis=0)
    
    # MLE for diagonal variances: σⱼ² = (1/n) * Σᵢ₌₁ⁿ (xᵢⱼ - μⱼ)² for j=1,2,3
    sigma_squared_hat = np.var(data, axis=0, ddof=0)
    
    # Create diagonal covariance matrix
    Sigma_diagonal = np.diag(sigma_squared_hat)
    
    return mu_hat, sigma_squared_hat, Sigma_diagonal


def analyze_bivariate_pairs(omega1_data, feature_names):
    """
    Analyze all possible pairs of features using bivariate MLE.
    """
    print(f"\n\n{'='*70}")
    print("PART 2: BIVARIATE ANALYSIS")
    print("Two-Dimensional Gaussian Data p(x) ~ N(μ, Σ)")
    print("=" * 70)
    
    # Define all possible pairs of features
    feature_pairs = [
        (0, 1, feature_names[0], feature_names[1]),  # x₁ and x₂
        (0, 2, feature_names[0], feature_names[2]),  # x₁ and x₃
        (1, 2, feature_names[1], feature_names[2])   # x₂ and x₃
    ]
    
    for pair_idx, (i, j, name_i, name_j) in enumerate(feature_pairs, 1):
        print(f"\nPAIR {pair_idx}: Features ({name_i}, {name_j})")
        print("-" * 60)
        
        # Extract the pair of features
        pair_data = omega1_data[:, [i, j]]
        
        print(f"Data points for ({name_i}, {name_j}):")
        for k, point in enumerate(pair_data, 1):
            print(f"  Point {k:2d}: [{point[0]:8.3f}, {point[1]:8.3f}]")
        print()
        
        # Compute MLE estimates
        mu_hat, Sigma_hat = mle_bivariate_gaussian(pair_data)
        
        print(f"MLE Results for ({name_i}, {name_j}):")
        print(f"  Mean vector μ̂:")
        print(f"    μ̂_{name_i} = {mu_hat[0]:10.6f}")
        print(f"    μ̂_{name_j} = {mu_hat[1]:10.6f}")
        print(f"    μ̂ = [{mu_hat[0]:10.6f}, {mu_hat[1]:10.6f}]")
        print()
        
        print(f"  Covariance matrix Σ̂:")
        print(f"    Σ̂ = [[{Sigma_hat[0,0]:10.6f}, {Sigma_hat[0,1]:10.6f}],")
        print(f"         [{Sigma_hat[1,0]:10.6f}, {Sigma_hat[1,1]:10.6f}]]")
        print()
        
        # Additional statistics
        correlation = Sigma_hat[0,1] / (np.sqrt(Sigma_hat[0,0]) * np.sqrt(Sigma_hat[1,1]))
        determinant = np.linalg.det(Sigma_hat)
        
        print(f"  Additional Statistics:")
        print(f"    Variance of {name_i}:        {Sigma_hat[0,0]:10.6f}")
        print(f"    Variance of {name_j}:        {Sigma_hat[1,1]:10.6f}")
        print(f"    Covariance:                  {Sigma_hat[0,1]:10.6f}")
        print(f"    Correlation coefficient:     {correlation:10.6f}")
        print(f"    Determinant of Σ̂:            {determinant:10.6f}")
        
        if pair_idx < len(feature_pairs):
            print("\n" + "=" * 70)
    
    # Summary table for bivariate results
    print(f"\n\n{'='*70}")
    print("SUMMARY TABLE - Bivariate MLE Results for Category ω₁")
    print("=" * 70)
    print(f"{'Pair':<12} {'μ̂₁':<12} {'μ̂₂':<12} {'σ̂₁₁²':<12} {'σ̂₂₂²':<12} {'σ̂₁₂':<12}")
    print("-" * 70)
    
    for pair_idx, (i, j, name_i, name_j) in enumerate(feature_pairs, 1):
        pair_data = omega1_data[:, [i, j]]
        mu_hat, Sigma_hat = mle_bivariate_gaussian(pair_data)
        pair_name = f"({name_i},{name_j})"
        print(f"{pair_name:<12} {mu_hat[0]:<12.6f} {mu_hat[1]:<12.6f} {Sigma_hat[0,0]:<12.6f} {Sigma_hat[1,1]:<12.6f} {Sigma_hat[0,1]:<12.6f}")
    
    print("=" * 70)


def analyze_trivariate_data(omega1_data, feature_names):
    """
    Analyze the full three-dimensional data using trivariate MLE.
    """
    print(f"\n\n{'='*75}")
    print("PART 3: TRIVARIATE ANALYSIS")
    print("Three-Dimensional Gaussian Data p(x) ~ N(μ, Σ)")
    print("Full 3D Analysis for Category ω₁")
    print("=" * 75)
    
    print(f"All Features: ({feature_names[0]}, {feature_names[1]}, {feature_names[2]})")
    print("-" * 65)
    
    # Use all three features
    trivariate_data = omega1_data  # All columns
    
    print(f"Three-dimensional data points:")
    for k, point in enumerate(trivariate_data, 1):
        print(f"  Point {k:2d}: [{point[0]:8.3f}, {point[1]:8.3f}, {point[2]:8.3f}]")
    print()
    
    # Compute MLE estimates for 3D data
    mu_hat, Sigma_hat = mle_trivariate_gaussian(trivariate_data)
    
    print(f"MLE Results for 3D Gaussian Distribution:")
    print(f"  Mean vector μ̂:")
    print(f"    μ̂_{feature_names[0]} = {mu_hat[0]:10.6f}")
    print(f"    μ̂_{feature_names[1]} = {mu_hat[1]:10.6f}")
    print(f"    μ̂_{feature_names[2]} = {mu_hat[2]:10.6f}")
    print(f"    μ̂ = [{mu_hat[0]:10.6f}, {mu_hat[1]:10.6f}, {mu_hat[2]:10.6f}]")
    print()
    
    print(f"  Covariance matrix Σ̂ (3×3):")
    print(f"    Σ̂ = [[{Sigma_hat[0,0]:10.6f}, {Sigma_hat[0,1]:10.6f}, {Sigma_hat[0,2]:10.6f}],")
    print(f"         [{Sigma_hat[1,0]:10.6f}, {Sigma_hat[1,1]:10.6f}, {Sigma_hat[1,2]:10.6f}],")
    print(f"         [{Sigma_hat[2,0]:10.6f}, {Sigma_hat[2,1]:10.6f}, {Sigma_hat[2,2]:10.6f}]]")
    print()
    
    # Calculate additional statistics
    determinant = np.linalg.det(Sigma_hat)
    trace = np.trace(Sigma_hat)
    eigenvalues = np.linalg.eigvals(Sigma_hat)
    
    # Calculate correlation matrix
    std_devs = np.sqrt(np.diag(Sigma_hat))
    correlation_matrix = Sigma_hat / np.outer(std_devs, std_devs)
    
    print(f"  Variance-Covariance Statistics:")
    print(f"    Variance of {feature_names[0]}:      {Sigma_hat[0,0]:10.6f}")
    print(f"    Variance of {feature_names[1]}:      {Sigma_hat[1,1]:10.6f}")
    print(f"    Variance of {feature_names[2]}:      {Sigma_hat[2,2]:10.6f}")
    print()
    print(f"    Covariance({feature_names[0]},{feature_names[1]}): {Sigma_hat[0,1]:10.6f}")
    print(f"    Covariance({feature_names[0]},{feature_names[2]}): {Sigma_hat[0,2]:10.6f}")
    print(f"    Covariance({feature_names[1]},{feature_names[2]}): {Sigma_hat[1,2]:10.6f}")
    print()
    
    print(f"  Correlation Matrix:")
    print(f"    R = [[{correlation_matrix[0,0]:8.6f}, {correlation_matrix[0,1]:8.6f}, {correlation_matrix[0,2]:8.6f}],")
    print(f"         [{correlation_matrix[1,0]:8.6f}, {correlation_matrix[1,1]:8.6f}, {correlation_matrix[1,2]:8.6f}],")
    print(f"         [{correlation_matrix[2,0]:8.6f}, {correlation_matrix[2,1]:8.6f}, {correlation_matrix[2,2]:8.6f}]]")
    print()
    
    print(f"  Matrix Properties:")
    print(f"    Determinant of Σ̂:         {determinant:15.6f}")
    print(f"    Trace of Σ̂:               {trace:15.6f}")
    print(f"    Eigenvalues:               [{eigenvalues[0]:8.6f}, {eigenvalues[1]:8.6f}, {eigenvalues[2]:8.6f}]")
    
    # Check if matrix is positive definite
    is_positive_definite = np.all(eigenvalues > 0)
    print(f"    Positive Definite:         {is_positive_definite}")
    
    # Volume of confidence ellipsoid (proportional to sqrt of determinant)
    ellipsoid_volume_factor = np.sqrt(determinant)
    print(f"    Ellipsoid Volume Factor:   {ellipsoid_volume_factor:15.6f}")
    
    print(f"\n{'='*75}")
    print("COMPREHENSIVE SUMMARY - All MLE Results for Category ω₁")
    print("=" * 75)
    
    # Summary of all analyses
    print(f"{'Analysis Type':<15} {'Features':<15} {'Mean(s)':<25} {'Variance(s)/Covariance':<30}")
    print("-" * 75)
    
    # Univariate summaries
    for i, name in enumerate(feature_names):
        feature_data = omega1_data[:, i]
        mu, sigma2 = mle_univariate_gaussian(feature_data)
        print(f"{'Univariate':<15} {name:<15} {mu:10.6f}{'':<15} {sigma2:15.6f}")
    
    print("-" * 75)
    
    # Bivariate summaries
    pairs = [(0,1,'x₁,x₂'), (0,2,'x₁,x₃'), (1,2,'x₂,x₃')]
    for i, j, pair_name in pairs:
        pair_data = omega1_data[:, [i, j]]
        mu_biv, Sigma_biv = mle_bivariate_gaussian(pair_data)
        det_biv = np.linalg.det(Sigma_biv)
        print(f"{'Bivariate':<15} {pair_name:<15} [{mu_biv[0]:6.3f},{mu_biv[1]:6.3f}]{'':<8} det={det_biv:10.6f}")
    
    print("-" * 75)
    
    # Trivariate summary
    print(f"{'Trivariate':<15} {'x₁,x₂,x₃':<15} [{mu_hat[0]:5.3f},{mu_hat[1]:5.3f},{mu_hat[2]:5.3f}]{'':<6} det={determinant:10.6f}")
    
    print("=" * 75)


def analyze_separable_model_omega2():
    """
    Analyze category ω₂ data using separable 3D Gaussian model where Σ = diag(σ₁², σ₂², σ₃²).
    """
    print(f"\n\n{'='*80}")
    print("PART 4: SEPARABLE MODEL ANALYSIS FOR CATEGORY ω₂")
    print("Three-Dimensional Separable Gaussian: Σ = diag(σ₁², σ₂², σ₃²)")
    print("=" * 80)
    
    # Data from the table for category ω₂ (10 data points, 3 features)
    omega2_data = np.array([
        [-0.4, 0.58, 0.089],      # point 1
        [-0.31, 0.27, -0.04],     # point 2
        [0.38, 0.055, -0.035],    # point 3
        [-0.15, 0.53, 0.011],     # point 4
        [-0.35, 0.47, 0.034],     # point 5
        [0.17, 0.69, 0.1],        # point 6
        [-0.011, 0.55, -0.18],    # point 7
        [-0.27, 0.61, 0.12],      # point 8
        [-0.065, 0.49, 0.0012],   # point 9
        [-0.12, 0.054, -0.063]    # point 10
    ])
    
    feature_names = ['x₁', 'x₂', 'x₃']
    
    print(f"Category ω₂ Data (n = {len(omega2_data)} points):")
    print("-" * 60)
    print(f"Three-dimensional data points:")
    for k, point in enumerate(omega2_data, 1):
        print(f"  Point {k:2d}: [{point[0]:8.3f}, {point[1]:8.3f}, {point[2]:8.3f}]")
    print()
    
    # Compute MLE estimates using separable model
    mu_hat, sigma_squared_hat, Sigma_diagonal = mle_separable_trivariate_gaussian(omega2_data)
    
    print(f"MLE Results for Separable 3D Gaussian (ω₂):")
    print("=" * 60)
    
    print(f"  Mean vector μ̂:")
    print(f"    μ̂₁ (x₁) = {mu_hat[0]:10.6f}")
    print(f"    μ̂₂ (x₂) = {mu_hat[1]:10.6f}")
    print(f"    μ̂₃ (x₃) = {mu_hat[2]:10.6f}")
    print(f"    μ̂ = [{mu_hat[0]:10.6f}, {mu_hat[1]:10.6f}, {mu_hat[2]:10.6f}]")
    print()
    
    print(f"  Diagonal Variances [σ₁², σ₂², σ₃²]:")
    print(f"    σ₁² (variance of x₁) = {sigma_squared_hat[0]:10.6f}")
    print(f"    σ₂² (variance of x₂) = {sigma_squared_hat[1]:10.6f}")
    print(f"    σ₃² (variance of x₃) = {sigma_squared_hat[2]:10.6f}")
    print(f"    [σ₁², σ₂², σ₃²] = [{sigma_squared_hat[0]:8.6f}, {sigma_squared_hat[1]:8.6f}, {sigma_squared_hat[2]:8.6f}]")
    print()
    
    print(f"  Standard Deviations [σ₁, σ₂, σ₃]:")
    sigma_hat = np.sqrt(sigma_squared_hat)
    print(f"    σ₁ (std dev of x₁) = {sigma_hat[0]:10.6f}")
    print(f"    σ₂ (std dev of x₂) = {sigma_hat[1]:10.6f}")
    print(f"    σ₃ (std dev of x₃) = {sigma_hat[2]:10.6f}")
    print()
    
    print(f"  Separable Covariance Matrix Σ = diag(σ₁², σ₂², σ₃²):")
    print(f"    Σ = [[{Sigma_diagonal[0,0]:10.6f}, {Sigma_diagonal[0,1]:10.6f}, {Sigma_diagonal[0,2]:10.6f}],")
    print(f"         [{Sigma_diagonal[1,0]:10.6f}, {Sigma_diagonal[1,1]:10.6f}, {Sigma_diagonal[1,2]:10.6f}],")
    print(f"         [{Sigma_diagonal[2,0]:10.6f}, {Sigma_diagonal[2,1]:10.6f}, {Sigma_diagonal[2,2]:10.6f}]]")
    print()
    
    # Calculate properties of diagonal matrix
    determinant_diag = np.prod(sigma_squared_hat)  # For diagonal matrix, det = product of diagonal elements
    trace_diag = np.sum(sigma_squared_hat)
    
    print(f"  Matrix Properties (Separable Model):")
    print(f"    Determinant of Σ:         {determinant_diag:15.6f}")
    print(f"    Trace of Σ:               {trace_diag:15.6f}")
    print(f"    Volume Factor (√det):     {np.sqrt(determinant_diag):15.6f}")
    print(f"    Independence Assumption:  All covariances = 0")
    print()
    
    # Compare with full covariance model for ω₂
    print(f"COMPARISON: Separable vs Full Covariance Model")
    print("-" * 60)
    
    # Compute full covariance matrix for comparison
    mu_full, Sigma_full = mle_trivariate_gaussian(omega2_data)
    det_full = np.linalg.det(Sigma_full)
    
    print(f"  Full Covariance Matrix (for comparison):")
    print(f"    Σ_full = [[{Sigma_full[0,0]:8.6f}, {Sigma_full[0,1]:8.6f}, {Sigma_full[0,2]:8.6f}],")
    print(f"              [{Sigma_full[1,0]:8.6f}, {Sigma_full[1,1]:8.6f}, {Sigma_full[1,2]:8.6f}],")
    print(f"              [{Sigma_full[2,0]:8.6f}, {Sigma_full[2,1]:8.6f}, {Sigma_full[2,2]:8.6f}]]")
    print()
    
    # Calculate off-diagonal elements to show what we're ignoring
    off_diag_elements = [
        (Sigma_full[0,1], 'Cov(x₁,x₂)'),
        (Sigma_full[0,2], 'Cov(x₁,x₃)'),
        (Sigma_full[1,2], 'Cov(x₂,x₃)')
    ]
    
    print(f"  Off-diagonal elements (ignored in separable model):")
    for cov_val, cov_name in off_diag_elements:
        print(f"    {cov_name:12} = {cov_val:10.6f}")
    print()
    
    print(f"  Model Comparison:")
    print(f"    Determinant (Separable):  {determinant_diag:15.6f}")
    print(f"    Determinant (Full):       {det_full:15.6f}")
    print(f"    Ratio (Sep/Full):         {determinant_diag/det_full:15.6f}")
    
    # Manual verification for one feature
    print(f"\n{'='*60}")
    print("MANUAL VERIFICATION - Feature x₁ of ω₂")
    print("=" * 60)
    
    x1_omega2 = omega2_data[:, 0]
    n = len(x1_omega2)
    
    print(f"x₁ data for ω₂: {x1_omega2}")
    print(f"n = {n}")
    
    # Manual calculation
    sum_x1 = np.sum(x1_omega2)
    mu1_manual = sum_x1 / n
    
    deviations = x1_omega2 - mu1_manual
    sum_dev_squared = np.sum(deviations ** 2)
    sigma1_squared_manual = sum_dev_squared / n
    
    print(f"\nμ₁ calculation:")
    print(f"  Σx₁ᵢ = {sum_x1:.6f}")
    print(f"  μ̂₁ = (1/{n}) × {sum_x1:.6f} = {mu1_manual:.6f}")
    
    print(f"\nσ₁² calculation:")
    print(f"  Deviations (x₁ᵢ - μ̂₁): {deviations}")
    print(f"  Σ(x₁ᵢ - μ̂₁)² = {sum_dev_squared:.6f}")
    print(f"  σ̂₁² = (1/{n}) × {sum_dev_squared:.6f} = {sigma1_squared_manual:.6f}")
    print(f"  σ̂₁ = √{sigma1_squared_manual:.6f} = {np.sqrt(sigma1_squared_manual):.6f}")
    
    print(f"\n{'='*80}")
    print("SUMMARY - Separable Model Results for ω₂")
    print("=" * 80)
    print(f"Assumption: Features are independent → Σ = diag(σ₁², σ₂², σ₃²)")
    print(f"Mean vector:     μ̂ = [{mu_hat[0]:7.4f}, {mu_hat[1]:7.4f}, {mu_hat[2]:7.4f}]")
    print(f"Diagonal variances: [σ₁², σ₂², σ₃²] = [{sigma_squared_hat[0]:.4f}, {sigma_squared_hat[1]:.4f}, {sigma_squared_hat[2]:.4f}]")
    print(f"Determinant:     det(Σ) = {determinant_diag:.6f}")
    print("=" * 80)


def compare_mean_estimates():
    """
    Compare mean estimates (μᵢ) calculated using different methods and explain differences.
    """
    print(f"\n\n{'='*85}")
    print("PART 5: COMPARISON OF MEAN ESTIMATES FROM DIFFERENT METHODS")
    print("Analysis of μᵢ values across Univariate, Bivariate, Trivariate, and Separable models")
    print("=" * 85)
    
    # Data for both categories
    omega1_data = np.array([
        [0.42, -0.087, 0.58],    # point 1
        [-0.2, -3.3, -3.4],     # point 2
        [1.3, -0.32, 1.7],      # point 3
        [0.39, 0.71, 0.23],     # point 4
        [-1.6, -5.3, -0.15],    # point 5
        [-0.029, 0.89, -4.7],   # point 6
        [-0.23, 1.9, 2.2],      # point 7
        [0.27, -0.3, -0.87],    # point 8
        [-1.9, 0.76, -2.1],     # point 9
        [0.87, -1.0, -2.6]      # point 10
    ])
    
    omega2_data = np.array([
        [-0.4, 0.58, 0.089],     # point 1
        [-0.31, 0.27, -0.04],    # point 2
        [0.38, 0.055, -0.035],   # point 3
        [-0.15, 0.53, 0.011],    # point 4
        [-0.35, 0.47, 0.034],    # point 5
        [0.17, 0.69, 0.1],       # point 6
        [-0.011, 0.55, -0.18],   # point 7
        [-0.27, 0.61, 0.12],     # point 8
        [-0.065, 0.49, 0.0012],  # point 9
        [-0.12, 0.054, -0.063]   # point 10
    ])
    
    feature_names = ['x₁', 'x₂', 'x₃']
    
    print("CATEGORY ω₁ - Mean Comparison")
    print("-" * 60)
    
    # Calculate means using different methods for ω₁
    print(f"{'Method':<20} {'μ₁ (x₁)':<12} {'μ₂ (x₂)':<12} {'μ₃ (x₃)':<12}")
    print("-" * 60)
    
    # 1. Univariate means
    mu1_univar = [mle_univariate_gaussian(omega1_data[:, i])[0] for i in range(3)]
    print(f"{'Univariate':<20} {mu1_univar[0]:<12.6f} {mu1_univar[1]:<12.6f} {mu1_univar[2]:<12.6f}")
    
    # 2. Bivariate means (from pairs)
    mu1_biv_12, _ = mle_bivariate_gaussian(omega1_data[:, [0, 1]])
    mu1_biv_13, _ = mle_bivariate_gaussian(omega1_data[:, [0, 2]])
    mu1_biv_23, _ = mle_bivariate_gaussian(omega1_data[:, [1, 2]])
    
    print(f"{'Bivariate (x₁,x₂)':<20} {mu1_biv_12[0]:<12.6f} {mu1_biv_12[1]:<12.6f} {'N/A':<12}")
    print(f"{'Bivariate (x₁,x₃)':<20} {mu1_biv_13[0]:<12.6f} {'N/A':<12} {mu1_biv_13[1]:<12.6f}")
    print(f"{'Bivariate (x₂,x₃)':<20} {'N/A':<12} {mu1_biv_23[0]:<12.6f} {mu1_biv_23[1]:<12.6f}")
    
    # 3. Trivariate means
    mu1_triv, _ = mle_trivariate_gaussian(omega1_data)
    print(f"{'Trivariate (full)':<20} {mu1_triv[0]:<12.6f} {mu1_triv[1]:<12.6f} {mu1_triv[2]:<12.6f}")
    
    print(f"\nCATEGORY ω₂ - Mean Comparison")
    print("-" * 60)
    
    print(f"{'Method':<20} {'μ₁ (x₁)':<12} {'μ₂ (x₂)':<12} {'μ₃ (x₃)':<12}")
    print("-" * 60)
    
    # 1. Univariate means for ω₂
    mu2_univar = [mle_univariate_gaussian(omega2_data[:, i])[0] for i in range(3)]
    print(f"{'Univariate':<20} {mu2_univar[0]:<12.6f} {mu2_univar[1]:<12.6f} {mu2_univar[2]:<12.6f}")
    
    # 2. Trivariate means for ω₂
    mu2_triv, _ = mle_trivariate_gaussian(omega2_data)
    print(f"{'Trivariate (full)':<20} {mu2_triv[0]:<12.6f} {mu2_triv[1]:<12.6f} {mu2_triv[2]:<12.6f}")
    
    # 3. Separable model means for ω₂
    mu2_sep, _, _ = mle_separable_trivariate_gaussian(omega2_data)
    print(f"{'Separable model':<20} {mu2_sep[0]:<12.6f} {mu2_sep[1]:<12.6f} {mu2_sep[2]:<12.6f}")
    
    # Detailed comparison with differences
    print(f"\nDETAILED COMPARISON - Differences Between Methods")
    print("=" * 85)
    
    print(f"CATEGORY ω₁ - Difference Analysis:")
    print("-" * 50)
    
    # Check if all means are identical for ω₁
    all_means_omega1 = {
        'Univariate': mu1_univar,
        'Bivariate_x1': [mu1_biv_12[0], mu1_biv_13[0], np.nan],
        'Bivariate_x2': [np.nan, mu1_biv_12[1], mu1_biv_23[0]],
        'Bivariate_x3': [np.nan, np.nan, mu1_biv_13[1], mu1_biv_23[1]],
        'Trivariate': mu1_triv.tolist()
    }
    
    print(f"Univariate vs Trivariate differences:")
    for i, name in enumerate(feature_names):
        diff = mu1_univar[i] - mu1_triv[i]
        print(f"  {name}: {diff:.10f} (Univariate - Trivariate)")
    
    print(f"\nBivariate vs Trivariate differences:")
    biv_means_x1 = [mu1_biv_12[0], mu1_biv_13[0]]
    biv_means_x2 = [mu1_biv_12[1], mu1_biv_23[0]]
    biv_means_x3 = [mu1_biv_13[1], mu1_biv_23[1]]
    
    print(f"  x₁: max diff = {max(abs(m - mu1_triv[0]) for m in biv_means_x1):.10f}")
    print(f"  x₂: max diff = {max(abs(m - mu1_triv[1]) for m in biv_means_x2):.10f}")
    print(f"  x₃: max diff = {max(abs(m - mu1_triv[2]) for m in biv_means_x3):.10f}")
    
    print(f"\nCATEGORY ω₂ - Difference Analysis:")
    print("-" * 50)
    
    print(f"Method comparison for ω₂:")
    for i, name in enumerate(feature_names):
        univar_vs_triv = mu2_univar[i] - mu2_triv[i]
        triv_vs_sep = mu2_triv[i] - mu2_sep[i]
        univar_vs_sep = mu2_univar[i] - mu2_sep[i]
        
        print(f"  {name}:")
        print(f"    Univariate - Trivariate: {univar_vs_triv:.10f}")
        print(f"    Trivariate - Separable:  {triv_vs_sep:.10f}")
        print(f"    Univariate - Separable:  {univar_vs_sep:.10f}")
    
    # Mathematical explanation
    print(f"\n{'='*85}")
    print("MATHEMATICAL EXPLANATION")
    print("=" * 85)
    
    print(f"""
WHY ARE THE MEAN ESTIMATES THE SAME OR DIFFERENT?

1. THEORETICAL FOUNDATION:
   The Maximum Likelihood Estimator for the mean of a multivariate Gaussian distribution is:
   
   μ̂ = (1/n) * Σᵢ₌₁ⁿ xᵢ
   
   This formula is INDEPENDENT of the covariance structure assumption.

2. KEY INSIGHT - MEAN INDEPENDENCE:
   The MLE for the mean vector does NOT depend on:
   - Whether features are assumed independent (separable model)
   - Whether we analyze features individually, in pairs, or together
   - The structure of the covariance matrix Σ
   
   The mean estimate ONLY depends on the marginal distribution of each feature.

3. ANALYSIS RESULTS:
   
   Category ω₁:
   - Univariate μ₁ = {mu1_univar[0]:.6f}
   - Bivariate μ₁  = {mu1_biv_12[0]:.6f} (from x₁,x₂) and {mu1_biv_13[0]:.6f} (from x₁,x₃)
   - Trivariate μ₁ = {mu1_triv[0]:.6f}
   - Maximum difference: {max(abs(mu1_univar[0] - mu1_triv[0]), abs(mu1_biv_12[0] - mu1_triv[0]), abs(mu1_biv_13[0] - mu1_triv[0])):.2e}
   
   Category ω₂:
   - Univariate μ₁ = {mu2_univar[0]:.6f}
   - Trivariate μ₁ = {mu2_triv[0]:.6f}
   - Separable μ₁  = {mu2_sep[0]:.6f}
   - Maximum difference: {max(abs(mu2_univar[0] - mu2_triv[0]), abs(mu2_triv[0] - mu2_sep[0]), abs(mu2_univar[0] - mu2_sep[0])):.2e}

4. CONCLUSION:
   ALL MEAN ESTIMATES ARE IDENTICAL (within numerical precision)
   
   This confirms the theoretical expectation that:
   - The sample mean is the MLE for μ regardless of dimensionality
   - Covariance assumptions do not affect mean estimation
   - μ̂ⱼ = (1/n) * Σᵢ₌₁ⁿ xᵢⱼ for each feature j, always

5. PRACTICAL IMPLICATION:
   Whether you use univariate, bivariate, trivariate, or separable models:
   - Mean estimates will be identical
   - Only covariance/variance estimates differ between models
   - Independence assumptions affect Σ, not μ
""")
    
    print("=" * 85)


def compare_variance_estimates():
    """
    Compare variance estimates (σᵢ²) calculated using different methods and explain differences.
    """
    print(f"\n\n{'='*90}")
    print("PART 6: COMPARISON OF VARIANCE ESTIMATES FROM DIFFERENT METHODS")
    print("Analysis of σᵢ² values across Univariate, Bivariate, Trivariate, and Separable models")
    print("=" * 90)
    
    # Data for both categories
    omega1_data = np.array([
        [0.42, -0.087, 0.58],    # point 1
        [-0.2, -3.3, -3.4],     # point 2
        [1.3, -0.32, 1.7],      # point 3
        [0.39, 0.71, 0.23],     # point 4
        [-1.6, -5.3, -0.15],    # point 5
        [-0.029, 0.89, -4.7],   # point 6
        [-0.23, 1.9, 2.2],      # point 7
        [0.27, -0.3, -0.87],    # point 8
        [-1.9, 0.76, -2.1],     # point 9
        [0.87, -1.0, -2.6]      # point 10
    ])
    
    omega2_data = np.array([
        [-0.4, 0.58, 0.089],     # point 1
        [-0.31, 0.27, -0.04],    # point 2
        [0.38, 0.055, -0.035],   # point 3
        [-0.15, 0.53, 0.011],    # point 4
        [-0.35, 0.47, 0.034],    # point 5
        [0.17, 0.69, 0.1],       # point 6
        [-0.011, 0.55, -0.18],   # point 7
        [-0.27, 0.61, 0.12],     # point 8
        [-0.065, 0.49, 0.0012],  # point 9
        [-0.12, 0.054, -0.063]   # point 10
    ])
    
    feature_names = ['x₁', 'x₂', 'x₃']
    
    print("CATEGORY ω₁ - Variance Comparison")
    print("-" * 70)
    
    # Calculate variances using different methods for ω₁
    print(f"{'Method':<25} {'σ₁² (x₁)':<15} {'σ₂² (x₂)':<15} {'σ₃² (x₃)':<15}")
    print("-" * 70)
    
    # 1. Univariate variances
    var1_univar = [mle_univariate_gaussian(omega1_data[:, i])[1] for i in range(3)]
    print(f"{'Univariate':<25} {var1_univar[0]:<15.6f} {var1_univar[1]:<15.6f} {var1_univar[2]:<15.6f}")
    
    # 2. Bivariate variances (diagonal elements from covariance matrices)
    _, Sigma1_biv_12 = mle_bivariate_gaussian(omega1_data[:, [0, 1]])
    _, Sigma1_biv_13 = mle_bivariate_gaussian(omega1_data[:, [0, 2]])
    _, Sigma1_biv_23 = mle_bivariate_gaussian(omega1_data[:, [1, 2]])
    
    print(f"{'Bivariate (x₁,x₂)':<25} {Sigma1_biv_12[0,0]:<15.6f} {Sigma1_biv_12[1,1]:<15.6f} {'N/A':<15}")
    print(f"{'Bivariate (x₁,x₃)':<25} {Sigma1_biv_13[0,0]:<15.6f} {'N/A':<15} {Sigma1_biv_13[1,1]:<15.6f}")
    print(f"{'Bivariate (x₂,x₃)':<25} {'N/A':<15} {Sigma1_biv_23[0,0]:<15.6f} {Sigma1_biv_23[1,1]:<15.6f}")
    
    # 3. Trivariate variances (diagonal elements)
    _, Sigma1_triv = mle_trivariate_gaussian(omega1_data)
    print(f"{'Trivariate (full)':<25} {Sigma1_triv[0,0]:<15.6f} {Sigma1_triv[1,1]:<15.6f} {Sigma1_triv[2,2]:<15.6f}")
    
    print(f"\nCATEGORY ω₂ - Variance Comparison")
    print("-" * 70)
    
    print(f"{'Method':<25} {'σ₁² (x₁)':<15} {'σ₂² (x₂)':<15} {'σ₃² (x₃)':<15}")
    print("-" * 70)
    
    # 1. Univariate variances for ω₂
    var2_univar = [mle_univariate_gaussian(omega2_data[:, i])[1] for i in range(3)]
    print(f"{'Univariate':<25} {var2_univar[0]:<15.6f} {var2_univar[1]:<15.6f} {var2_univar[2]:<15.6f}")
    
    # 2. Trivariate variances for ω₂
    _, Sigma2_triv = mle_trivariate_gaussian(omega2_data)
    print(f"{'Trivariate (full)':<25} {Sigma2_triv[0,0]:<15.6f} {Sigma2_triv[1,1]:<15.6f} {Sigma2_triv[2,2]:<15.6f}")
    
    # 3. Separable model variances for ω₂
    _, var2_sep, _ = mle_separable_trivariate_gaussian(omega2_data)
    print(f"{'Separable model':<25} {var2_sep[0]:<15.6f} {var2_sep[1]:<15.6f} {var2_sep[2]:<15.6f}")
    
    # Detailed comparison with differences
    print(f"\nDETAILED VARIANCE COMPARISON - Differences Between Methods")
    print("=" * 90)
    
    print(f"CATEGORY ω₁ - Variance Difference Analysis:")
    print("-" * 60)
    
    # Check differences for ω₁
    print(f"Univariate vs Trivariate variance differences:")
    for i, name in enumerate(feature_names):
        diff = var1_univar[i] - Sigma1_triv[i,i]
        print(f"  {name}: {diff:.12f} (Univariate - Trivariate)")
    
    print(f"\nBivariate vs Trivariate variance differences:")
    biv_var1_x1 = [Sigma1_biv_12[0,0], Sigma1_biv_13[0,0]]
    biv_var1_x2 = [Sigma1_biv_12[1,1], Sigma1_biv_23[0,0]]
    biv_var1_x3 = [Sigma1_biv_13[1,1], Sigma1_biv_23[1,1]]
    
    print(f"  x₁: differences = {[abs(v - Sigma1_triv[0,0]) for v in biv_var1_x1]}")
    print(f"      max diff = {max(abs(v - Sigma1_triv[0,0]) for v in biv_var1_x1):.12f}")
    print(f"  x₂: differences = {[abs(v - Sigma1_triv[1,1]) for v in biv_var1_x2]}")
    print(f"      max diff = {max(abs(v - Sigma1_triv[1,1]) for v in biv_var1_x2):.12f}")
    print(f"  x₃: differences = {[abs(v - Sigma1_triv[2,2]) for v in biv_var1_x3]}")
    print(f"      max diff = {max(abs(v - Sigma1_triv[2,2]) for v in biv_var1_x3):.12f}")
    
    print(f"\nCATEGORY ω₂ - Variance Difference Analysis:")
    print("-" * 60)
    
    print(f"Method comparison for ω₂:")
    for i, name in enumerate(feature_names):
        univar_vs_triv = var2_univar[i] - Sigma2_triv[i,i]
        triv_vs_sep = Sigma2_triv[i,i] - var2_sep[i]
        univar_vs_sep = var2_univar[i] - var2_sep[i]
        
        print(f"  {name}:")
        print(f"    Univariate - Trivariate: {univar_vs_triv:.12f}")
        print(f"    Trivariate - Separable:  {triv_vs_sep:.12f}")
        print(f"    Univariate - Separable:  {univar_vs_sep:.12f}")
    
    # Mathematical explanation
    print(f"\n{'='*90}")
    print("MATHEMATICAL EXPLANATION - VARIANCE ESTIMATES")
    print("=" * 90)
    
    print(f"""
WHY ARE THE VARIANCE ESTIMATES THE SAME OR DIFFERENT?

1. THEORETICAL FOUNDATION:
   For multivariate Gaussian MLE, the variance of feature j is:
   
   σⱼ² = (1/n) * Σᵢ₌₁ⁿ (xᵢⱼ - μ̂ⱼ)²
   
   This is the DIAGONAL element of the covariance matrix Σ̂.

2. KEY INSIGHT - DIAGONAL ELEMENT CONSISTENCY:
   The diagonal elements of Σ̂ are the SAME regardless of:
   - Dimensionality of the analysis (univariate, bivariate, trivariate)
   - Independence assumptions (separable vs full covariance)
   - Which other features are included in the analysis
   
   The diagonal elements represent the MARGINAL variances of each feature.

3. ANALYSIS RESULTS:
   
   Category ω₁ - Maximum variance differences:
   - x₁: max diff = {max(abs(var1_univar[0] - Sigma1_triv[0,0]), max(abs(v - Sigma1_triv[0,0]) for v in biv_var1_x1)):.2e}
   - x₂: max diff = {max(abs(var1_univar[1] - Sigma1_triv[1,1]), max(abs(v - Sigma1_triv[1,1]) for v in biv_var1_x2)):.2e}
   - x₃: max diff = {max(abs(var1_univar[2] - Sigma1_triv[2,2]), max(abs(v - Sigma1_triv[2,2]) for v in biv_var1_x3)):.2e}
   
   Category ω₂ - Maximum variance differences:
   - x₁: max diff = {max(abs(var2_univar[0] - Sigma2_triv[0,0]), abs(Sigma2_triv[0,0] - var2_sep[0]), abs(var2_univar[0] - var2_sep[0])):.2e}
   - x₂: max diff = {max(abs(var2_univar[1] - Sigma2_triv[1,1]), abs(Sigma2_triv[1,1] - var2_sep[1]), abs(var2_univar[1] - var2_sep[1])):.2e}
   - x₃: max diff = {max(abs(var2_univar[2] - Sigma2_triv[2,2]), abs(Sigma2_triv[2,2] - var2_sep[2]), abs(var2_univar[2] - var2_sep[2])):.2e}

4. WHY VARIANCES ARE IDENTICAL:
   
   a) MARGINAL VARIANCE PROPERTY:
      - Each σⱼ² represents the variance of feature j ALONE
      - This is independent of correlations with other features
      - Including/excluding other features doesn't change marginal variance
   
   b) MLE CONSISTENCY:
      - The MLE for variance is the sample variance
      - Sample variance of feature j is always (1/n) * Σᵢ(xᵢⱼ - x̄ⱼ)²
      - This formula is invariant to model dimensionality
   
   c) SEPARABLE MODEL EQUIVALENCE:
      - Separable model: Σ = diag(σ₁², σ₂², σ₃²)
      - Full model diagonal: Σ[j,j] = σⱼ²
      - Both compute the same marginal variance!

5. WHAT DIFFERS BETWEEN MODELS:
   
   SAME across all models:
   ✅ Diagonal elements (variances): σ₁², σ₂², σ₃²
   ✅ Mean estimates: μ₁, μ₂, μ₃
   
   DIFFERENT between models:
   ❌ Off-diagonal elements (covariances): σᵢⱼ where i≠j
   ❌ Determinant: det(Σ)
   ❌ Correlation coefficients: ρᵢⱼ
   ❌ Eigenvalues and eigenvectors

6. PRACTICAL IMPLICATIONS:
   
   a) COMPUTATIONAL EFFICIENCY:
      - Variance estimation: Univariate analysis sufficient
      - No need for multivariate analysis if only variances needed
   
   b) MODEL ROBUSTNESS:
      - Variance estimates robust to correlation assumptions
      - Independence assumption doesn't affect individual feature variability
   
   c) FEATURE SELECTION:
      - Individual feature variances unchanged by feature subset selection
      - Can estimate variances from any subset containing the feature

7. CONCLUSION:
   ALL VARIANCE ESTIMATES ARE IDENTICAL because:
   - Diagonal elements of covariance matrix are marginal variances
   - Marginal variances are independent of correlation structure
   - MLE for variance is always the sample variance of each feature
   - Independence assumptions only affect off-diagonal elements
""")
    
    # Summary table
    print(f"\n{'='*90}")
    print("SUMMARY - Variance Consistency Verification")
    print("=" * 90)
    
    print(f"RESULT: All variance estimates are IDENTICAL across different methods")
    print(f"REASON: Diagonal elements of covariance matrices represent marginal variances")
    print(f"IMPACT: Model choice affects correlations, NOT individual feature variances")
    
    print("=" * 90)


def main():
    """
    Main function to analyze category ω₁ data and compute MLE for each feature.
    """
    
    # Data from the table for category ω₁ (10 data points, 3 features)
    omega1_data = np.array([
        [0.42, -0.087, 0.58],    # point 1
        [-0.2, -3.3, -3.4],     # point 2
        [1.3, -0.32, 1.7],      # point 3
        [0.39, 0.71, 0.23],     # point 4
        [-1.6, -5.3, -0.15],    # point 5
        [-0.029, 0.89, -4.7],   # point 6
        [-0.23, 1.9, 2.2],      # point 7
        [0.27, -0.3, -0.87],    # point 8
        [-1.9, 0.76, -2.1],     # point 9
        [0.87, -1.0, -2.6]      # point 10
    ])
    
    print("Maximum Likelihood Estimation for Category ω₁")
    print("=" * 55)
    print(f"Number of data points: {len(omega1_data)}")
    print(f"Number of features: {omega1_data.shape[1]}")
    print()
    
    print("MLE Formulas:")
    print("UNIVARIATE: μ̂ = (1/n) * Σᵢ₌₁ⁿ xᵢ")
    print("           σ̂² = (1/n) * Σᵢ₌₁ⁿ (xᵢ - μ̂)²")
    print("BIVARIATE:  μ̂ = (1/n) * Σᵢ₌₁ⁿ xᵢ  (2D vector)")
    print("           Σ̂ = (1/n) * Σᵢ₌₁ⁿ (xᵢ - μ̂)(xᵢ - μ̂)ᵀ  (2×2 matrix)")
    print("TRIVARIATE: μ̂ = (1/n) * Σᵢ₌₁ⁿ xᵢ  (3D vector)")
    print("           Σ̂ = (1/n) * Σᵢ₌₁ⁿ (xᵢ - μ̂)(xᵢ - μ̂)ᵀ  (3×3 matrix)")
    print()
    
    # Feature names
    feature_names = ['x₁', 'x₂', 'x₃']
    
    print("PART 1: UNIVARIATE ANALYSIS")
    print("Individual Feature Analysis:")
    print("-" * 55)
    
    # Store results for summary table
    results = []
    
    for i, feature_name in enumerate(feature_names):
        feature_data = omega1_data[:, i]
        mu_hat, sigma_squared_hat = mle_univariate_gaussian(feature_data)
        sigma_hat = np.sqrt(sigma_squared_hat)
        
        results.append((feature_name, mu_hat, sigma_squared_hat, sigma_hat))
        
        print(f"\nFeature {feature_name}:")
        print(f"  Data points: {feature_data}")
        print(f"  μ̂ (mean estimate)     = {mu_hat:.6f}")
        print(f"  σ̂² (variance estimate) = {sigma_squared_hat:.6f}")
        print(f"  σ̂ (std dev estimate)   = {sigma_hat:.6f}")
    
    # Summary table
    print(f"\n{'='*55}")
    print("SUMMARY TABLE - Univariate MLE Results for Category ω₁")
    print("=" * 55)
    print(f"{'Feature':<10} {'μ̂ (Mean)':<15} {'σ̂² (Variance)':<15} {'σ̂ (Std Dev)':<15}")
    print("-" * 55)
    
    for feature_name, mu_hat, sigma_squared_hat, sigma_hat in results:
        print(f"{feature_name:<10} {mu_hat:<15.6f} {sigma_squared_hat:<15.6f} {sigma_hat:<15.6f}")
    
    print("=" * 55)
    
    # PART 2: BIVARIATE ANALYSIS
    analyze_bivariate_pairs(omega1_data, feature_names)
    
    # PART 3: TRIVARIATE ANALYSIS
    analyze_trivariate_data(omega1_data, feature_names)
    
    # PART 4: SEPARABLE MODEL ANALYSIS FOR ω₂
    analyze_separable_model_omega2()
    
    # PART 5: COMPARISON OF MEAN ESTIMATES
    compare_mean_estimates()
    
    # PART 6: COMPARISON OF VARIANCE ESTIMATES
    compare_variance_estimates()
    
    # Manual verification for one feature (x₁) to show step-by-step calculation
    print(f"\nManual Verification for Feature x₁:")
    print("-" * 40)
    
    x1_data = omega1_data[:, 0]
    n = len(x1_data)
    
    print(f"x₁ data: {x1_data}")
    print(f"n = {n}")
    
    # Step-by-step mean calculation
    sum_x1 = np.sum(x1_data)
    mu_manual = sum_x1 / n
    print(f"\nμ̂ calculation:")
    print(f"  Σx₁ᵢ = {sum_x1:.6f}")
    print(f"  μ̂ = (1/{n}) * {sum_x1:.6f} = {mu_manual:.6f}")
    
    # Step-by-step variance calculation
    deviations = x1_data - mu_manual
    deviations_squared = deviations ** 2
    sum_dev_squared = np.sum(deviations_squared)
    sigma2_manual = sum_dev_squared / n
    
    print(f"\nσ̂² calculation:")
    print(f"  (xᵢ - μ̂): {deviations}")
    print(f"  (xᵢ - μ̂)²: {deviations_squared}")
    print(f"  Σ(xᵢ - μ̂)² = {sum_dev_squared:.6f}")
    print(f"  σ̂² = (1/{n}) * {sum_dev_squared:.6f} = {sigma2_manual:.6f}")
    print(f"  σ̂ = √{sigma2_manual:.6f} = {np.sqrt(sigma2_manual):.6f}")


if __name__ == "__main__":
    main()