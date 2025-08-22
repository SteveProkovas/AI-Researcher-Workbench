# gaussian_integral.py
# A script exploring the Gaussian integral numerically and visually.
# Run with: python gaussian_integral.py [--dx STEP_SIZE] [--lim LIMIT]

import numpy as np
import math
import matplotlib.pyplot as plt
import argparse

def composite_simpson(y, x):
    """
    Compute the integral using Composite Simpson's Rule.
    
    Parameters:
    y: array of function values
    x: array of x points
    
    Returns:
    Integral estimate
    """
    n = len(x) - 1
    if n % 2 == 1:
        x = x[:-1]
        y = y[:-1]
        n = len(x) - 1
    dx = x[1] - x[0]
    S = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2])
    return dx * S / 3.0

def gaussian(x):
    """Standard Gaussian function: e^(-x²)"""
    return np.exp(-x**2)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Numerical evaluation of the Gaussian integral')
    parser.add_argument('--dx', type=float, default=0.001, 
                       help='Step size for numerical integration (default: 0.001)')
    parser.add_argument('--lim', type=float, default=6.0,
                       help='Integration limit (default: 6.0)')
    args = parser.parse_args()
    
    # Known analytic result
    analytic = math.sqrt(math.pi)
    print(f"Analytic result: ∫e^(-x²)dx from -∞ to ∞ = √π ≈ {analytic:.10f}")
    
    # Numerical integration
    dx = args.dx
    lim = args.lim
    x = np.arange(-lim, lim + dx, dx)
    y = gaussian(x)
    est = composite_simpson(y, x)
    
    # Error analysis
    abs_error = abs(est - analytic)
    rel_error = abs_error / analytic * 100
    
    print(f"Numerical estimate on [{-lim:.1f}, {lim:.1f}] with dx={dx}: {est:.10f}")
    print(f"Absolute error: {abs_error:.2e}")
    print(f"Relative error: {rel_error:.2e}%")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot the Gaussian curve
    plt.subplot(2, 1, 1)
    plt.plot(x, y, 'b-', linewidth=2, label='$f(x) = e^{-x^2}$')
    plt.fill_between(x, y, alpha=0.3, color='blue', label='Area under curve')
    
    # Add annotations
    plt.title('Gaussian Function and Its Integral', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add result textbox
    textstr = '\n'.join((
        f'Analytic: $\\sqrt{{\\pi}} \\approx {analytic:.7f}$',
        f'Numerical: ${est:.7f}$',
        f'Abs Error: ${abs_error:.2e}$',
        f'Rel Error: ${rel_error:.2e}\\%$'))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=props)
    
    # Convergence analysis (for fixed lim, varying dx)
    plt.subplot(2, 1, 2)
    dx_values = [0.1, 0.05, 0.01, 0.005, 0.001]
    errors = []
    
    for dx_val in dx_values:
        x_conv = np.arange(-lim, lim + dx_val, dx_val)
        y_conv = gaussian(x_conv)
        est_conv = composite_simpson(y_conv, x_conv)
        errors.append(abs(est_conv - analytic))
    
    plt.loglog(dx_values, errors, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Step Size (dx)', fontsize=12)
    plt.ylabel('Absolute Error', fontsize=12)
    plt.title('Convergence of Simpson\'s Rule', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    # Add a reference line showing O(dx⁴) convergence
    ref_dx = np.array(dx_values)
    ref_error = 0.1 * ref_dx**4  # Arbitrary scaling for visualization
    plt.loglog(ref_dx, ref_error, 'r--', label='O(dx$^4$) reference')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('gaussian_integral.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
