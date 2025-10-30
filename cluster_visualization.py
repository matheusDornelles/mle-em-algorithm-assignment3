"""
Cluster Visualization and Summary for EM Algorithm Results

This script provides additional analysis and visualization of the EM algorithm results
for estimating Gaussian parameters with missing data.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_cluster_visualization():
    """
    Create visualization of the estimated clusters and missing data analysis.
    """
    # Original ω₁ data
    omega1_data = np.array([
        [0.42, -0.087, 0.58],    # point 1
        [-0.2, -3.3, -3.4],     # point 2 - x₃ missing
        [1.3, -0.32, 1.7],      # point 3
        [0.39, 0.71, 0.23],     # point 4 - x₃ missing
        [-1.6, -5.3, -0.15],    # point 5
        [-0.029, 0.89, -4.7],   # point 6 - x₃ missing
        [-0.23, 1.9, 2.2],      # point 7
        [0.27, -0.3, -0.87],    # point 8 - x₃ missing
        [-1.9, 0.76, -2.1],     # point 9
        [0.87, -1.0, -2.6]      # point 10 - x₃ missing
    ])
    
    # EM estimated parameters (both strategies converge to same result)
    mu_estimated = np.array([-0.070900, -0.604700, 0.772558])
    sigma_estimated = np.array([
        [0.906177, 0.567782, 0.881340],
        [0.567782, 4.200715, 0.462137],
        [0.881340, 0.462137, 1.782672]
    ])
    
    # True parameters for comparison
    mu_true = np.mean(omega1_data, axis=0)
    sigma_true = np.cov(omega1_data.T, ddof=0)
    
    missing_indices = [1, 3, 5, 7, 9]  # 0-based indices for points 2,4,6,8,10
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 3D Scatter plot of original vs estimated
    ax1 = fig.add_subplot(2, 4, 1, projection='3d')
    
    # Plot complete data points
    complete_mask = np.ones(len(omega1_data), dtype=bool)
    complete_mask[missing_indices] = False
    
    ax1.scatter(omega1_data[complete_mask, 0], omega1_data[complete_mask, 1], 
                omega1_data[complete_mask, 2], c='blue', s=100, alpha=0.8, 
                label='Complete Data', marker='o')
    
    # Plot missing data points (with true x₃)
    ax1.scatter(omega1_data[missing_indices, 0], omega1_data[missing_indices, 1], 
                omega1_data[missing_indices, 2], c='red', s=100, alpha=0.8, 
                label='Missing x₃ (True)', marker='s')
    
    # Plot estimated cluster center
    ax1.scatter([mu_estimated[0]], [mu_estimated[1]], [mu_estimated[2]], 
                c='green', s=200, marker='*', label='EM Estimate')
    ax1.scatter([mu_true[0]], [mu_true[1]], [mu_true[2]], 
                c='orange', s=200, marker='*', label='True Mean')
    
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂') 
    ax1.set_zlabel('x₃')
    ax1.set_title('3D Data Points and Cluster Centers')
    ax1.legend()
    
    # 2. Feature projections
    ax2 = fig.add_subplot(2, 4, 2)
    ax2.scatter(omega1_data[complete_mask, 0], omega1_data[complete_mask, 1], 
                c='blue', s=100, alpha=0.8, label='Complete')
    ax2.scatter(omega1_data[missing_indices, 0], omega1_data[missing_indices, 1], 
                c='red', s=100, alpha=0.8, label='Missing x₃')
    ax2.scatter(mu_estimated[0], mu_estimated[1], c='green', s=200, marker='*', label='EM')
    ax2.scatter(mu_true[0], mu_true[1], c='orange', s=200, marker='*', label='True')
    ax2.set_xlabel('x₁')
    ax2.set_ylabel('x₂')
    ax2.set_title('x₁ vs x₂ Projection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Missing vs observed x₃ values
    ax3 = fig.add_subplot(2, 4, 3)
    true_x3_missing = omega1_data[missing_indices, 2]
    estimated_x3_missing = np.full(len(missing_indices), mu_estimated[2])
    
    x_pos = np.arange(len(missing_indices))
    width = 0.35
    
    ax3.bar(x_pos - width/2, true_x3_missing, width, label='True x₃', alpha=0.8, color='red')
    ax3.bar(x_pos + width/2, estimated_x3_missing, width, label='Estimated x₃', alpha=0.8, color='green')
    
    ax3.set_xlabel('Missing Data Points')
    ax3.set_ylabel('x₃ Value')
    ax3.set_title('Missing x₃: True vs Estimated')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(['Pt 2', 'Pt 4', 'Pt 6', 'Pt 8', 'Pt 10'])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Parameter comparison
    ax4 = fig.add_subplot(2, 4, 4)
    params = ['μ₁', 'μ₂', 'μ₃']
    true_means = mu_true
    est_means = mu_estimated
    
    x_pos = np.arange(len(params))
    ax4.bar(x_pos - width/2, true_means, width, label='True', alpha=0.8, color='orange')
    ax4.bar(x_pos + width/2, est_means, width, label='EM Estimate', alpha=0.8, color='green')
    
    ax4.set_xlabel('Parameters')
    ax4.set_ylabel('Value')
    ax4.set_title('Mean Parameters: True vs Estimated')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(params)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Variance comparison
    ax5 = fig.add_subplot(2, 4, 5)
    true_vars = np.diag(sigma_true)
    est_vars = np.diag(sigma_estimated)
    params_var = ['σ₁²', 'σ₂²', 'σ₃²']
    
    ax5.bar(x_pos - width/2, true_vars, width, label='True', alpha=0.8, color='orange')
    ax5.bar(x_pos + width/2, est_vars, width, label='EM Estimate', alpha=0.8, color='green')
    
    ax5.set_xlabel('Parameters')
    ax5.set_ylabel('Variance')
    ax5.set_title('Variance Parameters: True vs Estimated')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(params_var)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Correlation heatmaps
    ax6 = fig.add_subplot(2, 4, 6)
    
    # True correlation
    D_true = np.diag(1.0 / np.sqrt(np.diag(sigma_true)))
    corr_true = np.dot(np.dot(D_true, sigma_true), D_true)
    
    im1 = ax6.imshow(corr_true, cmap='RdBu_r', vmin=-1, vmax=1)
    ax6.set_title('True Correlation Matrix')
    ax6.set_xticks([0, 1, 2])
    ax6.set_yticks([0, 1, 2])
    ax6.set_xticklabels(['x₁', 'x₂', 'x₃'])
    ax6.set_yticklabels(['x₁', 'x₂', 'x₃'])
    
    # Add correlation values
    for i in range(3):
        for j in range(3):
            ax6.text(j, i, f'{corr_true[i,j]:.3f}', ha='center', va='center', color='black')
    
    plt.colorbar(im1, ax=ax6, shrink=0.6)
    
    # 7. Estimated correlation 
    ax7 = fig.add_subplot(2, 4, 7)
    
    D_est = np.diag(1.0 / np.sqrt(np.diag(sigma_estimated)))
    corr_est = np.dot(np.dot(D_est, sigma_estimated), D_est)
    
    im2 = ax7.imshow(corr_est, cmap='RdBu_r', vmin=-1, vmax=1)
    ax7.set_title('EM Estimated Correlation Matrix')
    ax7.set_xticks([0, 1, 2])
    ax7.set_yticks([0, 1, 2])
    ax7.set_xticklabels(['x₁', 'x₂', 'x₃'])
    ax7.set_yticklabels(['x₁', 'x₂', 'x₃'])
    
    # Add correlation values
    for i in range(3):
        for j in range(3):
            ax7.text(j, i, f'{corr_est[i,j]:.3f}', ha='center', va='center', color='black')
    
    plt.colorbar(im2, ax=ax7, shrink=0.6)
    
    # 8. Error analysis
    ax8 = fig.add_subplot(2, 4, 8)
    
    # Calculate errors
    mean_errors = np.abs(mu_estimated - mu_true)
    var_errors = np.abs(np.diag(sigma_estimated) - np.diag(sigma_true))
    
    x_pos = np.arange(3)
    ax8.bar(x_pos - width/2, mean_errors, width, label='Mean Error |μ̂ - μ|', alpha=0.8, color='red')
    ax8.bar(x_pos + width/2, var_errors, width, label='Variance Error |σ̂² - σ²|', alpha=0.8, color='blue')
    
    ax8.set_xlabel('Features')
    ax8.set_ylabel('Absolute Error')
    ax8.set_title('Parameter Estimation Errors')
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(['Feature 1', 'Feature 2', 'Feature 3'])
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('cluster_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
    print("\nComprehensive cluster analysis saved as 'cluster_analysis_comprehensive.png'")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("CLUSTER ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nEstimated Cluster Center:")
    print(f"  μ̂ = [{mu_estimated[0]:8.4f}, {mu_estimated[1]:8.4f}, {mu_estimated[2]:8.4f}]")
    
    print(f"\nTrue Cluster Center:")
    print(f"  μ = [{mu_true[0]:8.4f}, {mu_true[1]:8.4f}, {mu_true[2]:8.4f}]")
    
    print(f"\nEstimation Errors:")
    for i, (true_val, est_val) in enumerate(zip(mu_true, mu_estimated)):
        error = abs(est_val - true_val)
        print(f"  |μ̂{i+1} - μ{i+1}| = {error:8.4f}")
    
    print(f"\nCluster Spread (Standard Deviations):")
    est_stds = np.sqrt(np.diag(sigma_estimated))
    true_stds = np.sqrt(np.diag(sigma_true))
    for i, (true_std, est_std) in enumerate(zip(true_stds, est_stds)):
        print(f"  True σ{i+1} = {true_std:6.4f}, Estimated σ{i+1} = {est_std:6.4f}")
    
    print(f"\nMissing Data Impact:")
    print(f"  x₃ estimation error: {abs(mu_estimated[2] - mu_true[2]):8.4f}")
    print(f"  x₃ variance underestimation: {abs(np.diag(sigma_estimated)[2] - np.diag(sigma_true)[2]):8.4f}")
    
    mse = np.mean((mu_estimated - mu_true)**2)
    print(f"\nOverall MSE (Mean): {mse:10.6f}")
    
    plt.show()

if __name__ == "__main__":
    create_cluster_visualization()