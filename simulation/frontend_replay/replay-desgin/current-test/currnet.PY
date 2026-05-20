#!/usr/bin/env python3
"""
Replay File Analysis Tool (Production-Ready)

This tool embodies rigorous statistical standards for traffic dynamics analysis.
It prioritizes mechanistic interpretation, penalizes overfitting via AIC/BIC,
and implements streaming architecture for scalability.

Usage:
    python main.py <replay_file> [--model TYPE] [--compare] [--plot] [--dynamics]

Examples:
    python main.py replay.txt --model logistic
    python main.py replay.txt --compare          # Compare all models
    python main.py replay.txt --dynamics         # Traffic dynamics analysis
    python main.py replay.txt --plot             # Show visualization
"""

import sys
import argparse
import math

# Check for numpy
try:
    import numpy as np
except ImportError:
    print("Error: NumPy is required. Please install it via 'pip install numpy'.")
    sys.exit(1)

# Check for scipy (required for non-linear models)
try:
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: SciPy not found. Exponential and Logistic models will be disabled.")

# Visualization (Optional)
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: Matplotlib not found. Visualization disabled.")

# ----------------------------------------------------------------------
# 1. Scalable Data Architecture (Streaming)
# ----------------------------------------------------------------------
def get_vehicle_counts_stream(filename):
    """
    Generator that streams vehicle counts from the file.
    This avoids loading the entire dataset (positions, angles, etc.) into memory,
    adhering to scalable architecture requirements for large replay files.
    
    Yields:
        int: Vehicle count for each frame
    """
    try:
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue
                
                # Basic validation: must contain separator
                if ';' not in line: continue
                
                # We only need the vehicle part (before semicolon) for counts
                veh_str = line.split(';')[0]
                
                # Count entries: vehicles are comma-separated
                # Filter empty strings that result from trailing commas
                count = sum(1 for v in veh_str.split(',') if v.strip())
                
                yield count
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Warning: Error parsing line {line_num}: {e}")

def parse_full_data(filename):
    """
    Parses full data (coordinates, lights, angles, etc.). 
    Use only if detailed spatial analysis is needed.
    For growth modeling, get_vehicle_counts_stream is preferred.
    
    Returns:
        tuple: (counts array, frames list)
    """
    frames = []
    counts = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ';' not in line: continue
            veh_str, tl_str = line.split(';')
            
            vehicles = []
            for entry in veh_str.split(','):
                if not entry: continue
                tokens = entry.strip().split()
                if len(tokens) != 7: continue
                vehicles.append({
                    'x': float(tokens[0]), 'y': float(tokens[1]),
                    'angle': float(tokens[2]), 'name': tokens[3],
                    'status': int(tokens[4]), 'length': float(tokens[5]),
                    'width': float(tokens[6])
                })
            
            traffic_lights = {}
            for entry in tl_str.split(','):
                if not entry: continue
                tokens = entry.strip().split()
                if len(tokens) != 4: continue
                traffic_lights[tokens[0]] = tokens[1:4]

            frames.append({'vehicles': vehicles, 'traffic_lights': traffic_lights})
            counts.append(len(vehicles))
    return np.array(counts), frames

# ----------------------------------------------------------------------
# 2. Statistical Framework (AIC/BIC)
# ----------------------------------------------------------------------
def calculate_information_criteria(n, k, rss):
    """
    Calculates AIC and BIC to penalize model complexity.
    Addresses the limitation of relying solely on R-squared.
    
    Args:
        n: Number of data points
        k: Number of model parameters
        rss: Residual sum of squares
        
    Returns:
        tuple: (AIC, BIC) values
    """
    if rss <= 0 or n <= k:
        return float('inf'), float('inf')
    
    # AIC = n * ln(RSS/n) + 2k
    aic = n * np.log(rss / n) + 2 * k
    # BIC = n * ln(RSS/n) + k * ln(n)
    bic = n * np.log(rss / n) + k * np.log(n)
    return aic, bic

def calculate_stats(n, k, rss, y_true, y_pred):
    """
    Calculates R-squared, AIC, and BIC.
    
    Args:
        n: Number of data points
        k: Number of parameters
        rss: Residual sum of squares
        y_true: Observed values
        y_pred: Predicted values
    """
    # R-squared
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (rss / ss_tot) if ss_tot != 0 else 1.0

    aic, bic = calculate_information_criteria(n, k, rss)
        
    return r2, aic, bic

# ----------------------------------------------------------------------
# 3. Mechanistic & Empirical Modeling
# ----------------------------------------------------------------------
def fit_logistic_mechanistic(x, y):
    """
    Fits Logistic Model: y = L / (1 + e^(-k(x - x0)))
    Provides physical interpretation of parameters (Carrying Capacity, Inflection).
    
    Returns:
        dict: Model results with interpretation
    """
    if not HAS_SCIPY: return {"error": "SciPy required."}
    
    func = lambda x, L, k, x0: L / (1 + np.exp(-k * (x - x0)))
    
    # Robust initial guesses
    L_guess = max(y) * 1.05
    x0_guess = x[len(y)//2] 
    k_guess = 0.01
    
    try:
        popt, pcov = curve_fit(func, x, y, p0=[L_guess, k_guess, x0_guess], maxfev=10000)
        L, k, x0 = popt
        y_pred = func(x, L, k, x0)
        rss = np.sum((y - y_pred) ** 2)
        
        r2 = 1 - (rss / np.sum((y - np.mean(y)) ** 2))
        aic, bic = calculate_information_criteria(len(x), 3, rss)
        
        return {
            "type": "Logistic (Mechanistic)",
            "equation": f"y = {L:.2f} / (1 + e^(-{k:.4f}(x - {x0:.2f})))",
            "params": {"L": L, "k": k, "x0": x0},
            "r_squared": r2, "aic": aic, "bic": bic,
            "interpretation": f"Saturation (Carrying Capacity): {L:.1f} vehicles. Inflection point: frame {x0:.1f}."
        }
    except Exception as e:
        return {"error": f"Logistic fit failed: {e}"}

def fit_polynomial_empirical(x, y, degree=2):
    """
    Fits Polynomial Model: y = ax^2 + bx + c
    Represents empirical fit without explicit physical mechanism assumptions.
    
    Args:
        x: Independent variable
        y: Dependent variable
        degree: Polynomial degree
        
    Returns:
        dict: Model results with interpretation
    """
    coefs = np.polyfit(x, y, degree)
    p = np.poly1d(coefs)
    y_pred = p(x)
    rss = np.sum((y - y_pred) ** 2)
    
    r2 = 1 - (rss / np.sum((y - np.mean(y)) ** 2))
    k = degree + 1
    aic, bic = calculate_information_criteria(len(x), k, rss)
    
    # Formatting equation
    terms = []
    for i, c in enumerate(coefs):
        power = degree - i
        if abs(c) < 1e-4: continue
        if power == 0: terms.append(f"{c:.4f}")
        elif power == 1: terms.append(f"{c:.4f}*x")
        else: terms.append(f"{c:.4f}*x^{power}")
        
    eq = "y = " + " + ".join(terms) if terms else "y = 0"
        
    return {
        "type": f"Polynomial (Empirical, deg {degree})",
        "equation": eq,
        "params": coefs.tolist(),
        "r_squared": r2, "aic": aic, "bic": bic,
        "interpretation": "Empirical curvature fit. No explicit physical saturation limit implied."
    }

def fit_exponential_model(x, y):
    """Fits y = a * exp(b * x)"""
    if not HAS_SCIPY:
        return {"error": "SciPy required for exponential fit."}
    
    # Model: y = a * e^(bx)
    func = lambda x, a, b: a * np.exp(b * x)
    
    # Initial guess: linear regression on log(y)
    mask = y > 0
    if np.sum(mask) < 2:
        return {"error": "Not enough positive data points for exponential fit."}
        
    p_guess = np.polyfit(x[mask], np.log(y[mask]), 1)
    a0 = np.exp(p_guess[1])
    b0 = p_guess[0]
    
    try:
        popt, _ = curve_fit(func, x, y, p0=[a0, b0], maxfev=10000)
        a, b = popt
        y_pred = func(x, a, b)
        rss = np.sum((y - y_pred) ** 2)
        r2, aic, bic = calculate_stats(len(x), 2, rss, y, y_pred)
        
        eq = f"y = {a:.4f} * e^({b:.4f} * x)"
        return {
            "type": "Exponential",
            "equation": eq,
            "r_squared": r2,
            "aic": aic,
            "bic": bic,
            "params": [a, b]
        }
    except Exception as e:
        return {"error": f"Exponential fit failed: {str(e)}"}

# ----------------------------------------------------------------------
# 4. Analysis Functions
# ----------------------------------------------------------------------
def analyze_replay_growth(filename):
    """Basic growth statistics."""
    counts, frames = parse_full_data(filename)
    if frames is None or not frames:
        return {"error": "No frames found or file not found."}

    n = len(counts)
    avg = sum(counts) / n if n else 0
    return {
        "frame_count": n,
        "counts": counts.tolist(),
        "average": avg,
        "min": min(counts),
        "max": max(counts)
    }

def run_comparison(x, y):
    """Runs all available models and compares them."""
    results = []
    
    # 1. Linear (Poly deg 1)
    res_lin = fit_polynomial_empirical(x, y, 1)
    if "error" not in res_lin: results.append(res_lin)
    
    # 2. Quadratic (Poly deg 2)
    res_quad = fit_polynomial_empirical(x, y, 2)
    if "error" not in res_quad: results.append(res_quad)
    
    # 3. Exponential
    if HAS_SCIPY:
        res_exp = fit_exponential_model(x, y)
        if "error" not in res_exp: results.append(res_exp)
        
        # 4. Logistic (Mechanistic)
        res_log = fit_logistic_mechanistic(x, y)
        if "error" not in res_log: results.append(res_log)
            
    # Sort by AIC (lower is better)
    results.sort(key=lambda k: k['aic'])
    
    return results

# ----------------------------------------------------------------------
# 5. Traffic Dynamics Analysis Functions
# ----------------------------------------------------------------------
def analyze_traffic_dynamics(vehicle_counts, window_size=10, iqr_factor=1.5, peak_prominence=None):
    """
    Performs comprehensive traffic analysis: Growth, Congestion, and Trend Prediction.
    
    Args:
        vehicle_counts (list/array): Raw vehicle counts per frame.
        window_size (int): Window size for rolling average and regression.
        iqr_factor (float): Sensitivity for significant growth detection (Default 1.5).
        peak_prominence (float): Minimum prominence for congestion peaks. 
                                 If None, defaults to 1 standard deviation.
                                 
    Returns:
        dict: Dictionary containing all computed arrays and indices.
    """
    counts = np.array(vehicle_counts)
    n_frames = len(counts)
    
    # ---- 1. Growth Analysis (Robust) ----
    growth_rates = np.diff(counts, prepend=0)
    
    # Relative Growth: (change / previous_count)
    relative_growth = np.zeros(n_frames)
    prev_counts = counts[:-1]
    valid_mask = prev_counts > 0
    
    relative_growth[1:][valid_mask] = growth_rates[1:][valid_mask] / prev_counts[valid_mask]

    # Significant Growth Detection (using IQR)
    positive_growth = growth_rates[growth_rates > 0]
    
    if len(positive_growth) > 0:
        q75, q25 = np.percentile(positive_growth, [75, 25])
        iqr = q75 - q25
        threshold = q75 + (iqr_factor * iqr)
        significant_growth_frames = np.where(growth_rates > threshold)[0].tolist()
    else:
        threshold = 0
        significant_growth_frames = []

    # Cumulative Growth
    cumulative_growth = np.cumsum(growth_rates)

    # ---- 2. Congestion Detection (Improved Peaks) ----
    height_th = np.mean(counts) + np.std(counts)
    
    if peak_prominence is None:
        peak_prominence = np.std(counts)
        
    peaks, _ = find_peaks(
        counts, 
        height=height_th, 
        prominence=peak_prominence,
        distance=window_size
    )

    # ---- 3. Trend Smoothing (Rolling Average) ----
    kernel = np.ones(window_size) / window_size
    rolling_avg = np.convolve(counts, kernel, mode='same')

    # ---- 4. Trend Prediction (Rolling Linear Regression) ----
    predicted_trend = np.full(n_frames, np.nan)
    
    if window_size >= 2:
        for i in range(n_frames - window_size + 1):
            window_counts = counts[i : i + window_size]
            window_x = np.arange(window_size)
            
            coefs = np.polyfit(window_x, window_counts, 1)
            slope, intercept = coefs[0], coefs[1]
            
            next_val = slope * window_size + intercept
            predicted_trend[i + window_size - 1] = next_val

    return {
        "growth_rates": growth_rates,
        "relative_growth": relative_growth,
        "cumulative_growth": cumulative_growth,
        "significant_growth_frames": significant_growth_frames,
        "congestion_peaks": peaks,
        "rolling_average": rolling_avg,
        "predicted_trend": predicted_trend
    }

def run_traffic_dynamics_analysis(x, y, window_size=10, iqr_factor=1.5):
    """Run the comprehensive traffic dynamics analysis and print results."""
    results = analyze_traffic_dynamics(y, window_size=window_size, iqr_factor=iqr_factor)
    
    print(f"\nTraffic Dynamics Analysis Results")
    print("=" * 60)
    print(f"Total Frames: {len(y)}")
    print(f"Window Size: {window_size}")
    print(f"IQR Factor: {iqr_factor}")
    print(f"\nGrowth Statistics:")
    print(f"  - Mean Growth Rate: {np.mean(results['growth_rates']):.2f}")
    print(f"  - Max Growth Rate: {np.max(results['growth_rates']):.2f}")
    print(f"  - Min Growth Rate: {np.min(results['growth_rates']):.2f}")
    print(f"  - Significant Growth Frames: {results['significant_growth_frames']}")
    print(f"\nCongestion Detection:")
    print(f"  - Peak Frames (Congestion): {results['congestion_peaks'].tolist()}")
    print(f"  - Number of Peaks: {len(results['congestion_peaks'])}")
    print(f"\nTrend Prediction:")
    valid_preds = results['predicted_trend'][~np.isnan(results['predicted_trend'])]
    if len(valid_preds) > 0:
        print(f"  - Average Predicted Value: {np.mean(valid_preds):.2f}")
        print(f"  - First 5 Predictions: {valid_preds[:5].tolist()}")
    
    return results

# ----------------------------------------------------------------------
# 6. Visualization
# ----------------------------------------------------------------------
def plot_results(x, y, results, peaks):
    """Generates interactive plots for model comparison and residuals."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not installed. Skipping plot.")
        return

    plt.figure(figsize=(12, 8))
    
    # Plot 1: Data & Fits
    plt.subplot(2, 1, 1)
    plt.scatter(x, y, s=5, alpha=0.5, label='Observed Data', color='gray')
    
    colors = ['blue', 'red', 'green', 'orange']
    for i, res in enumerate(results):
        if "error" in res: continue
        # Reconstruct y_pred for plotting
        if "Logistic" in res['type']:
            L, k, x0 = res['params']['L'], res['params']['k'], res['params']['x0']
            y_fit = L / (1 + np.exp(-k * (x - x0)))
        elif "Exponential" in res['type']:
            a, b = res['params'][0], res['params'][1]
            y_fit = a * np.exp(b * x)
        else:
            coefs = res['params']
            p = np.poly1d(coefs)
            y_fit = p(x)
            
        plt.plot(x, y_fit, linewidth=2, label=f"{res['type']} (AIC: {res['aic']:.0f})")

    if peaks.size > 0:
        plt.plot(x[peaks], y[peaks], "x", color='black', markersize=10, label='Congestion Peaks')
        
    plt.title("Traffic Growth Model Comparison")
    plt.xlabel("Frame")
    plt.ylabel("Vehicle Count")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Residuals
    plt.subplot(2, 1, 2)
    best_res = results[0] # Assuming sorted by AIC
    if "Logistic" in best_res['type']:
        L, k, x0 = best_res['params']['L'], best_res['params']['k'], best_res['params']['x0']
        y_fit = L / (1 + np.exp(-k * (x - x0)))
    elif "Exponential" in best_res['type']:
        a, b = best_res['params'][0], best_res['params'][1]
        y_fit = a * np.exp(b * x)
    else:
        p = np.poly1d(best_res['params'])
        y_fit = p(x)
        
    residuals = y - y_fit
    plt.scatter(x, residuals, s=5, color='purple')
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f"Residuals of Best Model ({best_res['type']})")
    plt.xlabel("Frame")
    plt.ylabel("Residual (Observed - Predicted)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------
# Main CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Advanced Traffic Dynamics Analysis - Production Ready",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py replay.txt --model linear
    python main.py replay.txt --model exponential
    python main.py replay.txt --model logistic
    python main.py replay.txt --compare          # Compare all models
    python main.py replay.txt --dynamics         # Traffic dynamics analysis
    python main.py replay.txt --plot             # Show interactive visualization
        """
    )
    parser.add_argument("filename", help="Path to the replay file")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model", choices=['linear', 'exponential', 'logistic'], 
                       help="Select a specific model to fit")
    group.add_argument("--compare", action="store_true", 
                       help="Compare all available models (Linear, Quadratic, Exponential, Logistic)")
    group.add_argument("--dynamics", action="store_true", 
                       help="Run traffic dynamics analysis (Growth, Congestion, Trend)")
    
    parser.add_argument("--degree", type=int, default=2, 
                        help="Polynomial degree if 'linear' is chosen (default: 2)")
    parser.add_argument("--growth", action="store_true", help="Show basic growth stats only")
    parser.add_argument("--window", type=int, default=10, 
                        help="Window size for traffic dynamics analysis (default: 10)")
    parser.add_argument("--iqr-factor", type=float, default=1.5, 
                        help="IQR factor for significant growth detection (default: 1.5)")
    parser.add_argument("--plot", action="store_true", 
                        help="Show interactive visualization (requires matplotlib)")
    
    args = parser.parse_args()

    # 1. Basic Growth Stats
    if args.growth:
        res = analyze_replay_growth(args.filename)
        if "error" in res:
            print("Error:", res["error"])
            sys.exit(1)
        print("Vehicle Growth Statistics")
        print("=========================")
        print(f"Total Frames: {res['frame_count']}")
        print(f"Average Vehicles: {res['average']:.2f}")
        print(f"Min Vehicles: {res['min']}")
        print(f"Max Vehicles: {res['max']}")
        sys.exit(0)

    # 2. Data Ingestion (Scalable Streaming)
    print(f"Processing {args.filename}...")
    # Use streaming for memory efficiency
    counts = np.array(list(get_vehicle_counts_stream(args.filename)))
    x = np.arange(len(counts))
    y = counts

    if len(y) == 0:
        print("Error: No data parsed.")
        return

    # 3. Congestion Detection (for visualization and dynamics)
    peaks = np.array([])
    if HAS_SCIPY:
        height = np.mean(y) + np.std(y)
        peaks, _ = find_peaks(y, height=height, prominence=np.std(y))
        print(f"Detected {len(peaks)} congestion peaks.")

    # 4. Traffic Dynamics Analysis Mode
    if args.dynamics:
        run_traffic_dynamics_analysis(x, y, window_size=args.window, iqr_factor=args.iqr_factor)
        sys.exit(0)

    # 5. Model Comparison Mode
    if args.compare:
        print(f"\nModel Comparison for '{args.filename}'")
        print("=" * 80)
        results = run_comparison(x, y)
        
        if not results:
            print("No models could be fitted.")
            sys.exit(1)
            
        print(f"{'Model Type':<30} | {'AIC':<10} | {'BIC':<10} | {'R²':<8}")
        print("-" * 70)
        for r in results:
            print(f"{r['type']:<30} | {r['aic']:<10.1f} | {r['bic']:<10.1f} | {r['r_squared']:<8.4f}")
        
        print("\n" + "=" * 80)
        print("Recommendation:")
        best = results[0]
        print(f"Best Model: {best['type']} (Lowest AIC)")
        print(f"Equation: {best['equation']}")
        if "interpretation" in best:
            print(f"Insight: {best['interpretation']}")
        
        # Visualization
        if args.plot:
            plot_results(x, y, results, peaks)
        
        sys.exit(0)

    # 6. Single Model Mode
    target_model = args.model
    if not target_model:
        target_model = 'linear' 

    result = None
    if target_model == 'linear':
        result = fit_polynomial_empirical(x, y, args.degree)
    elif target_model == 'exponential':
        result = fit_exponential_model(x, y)
    elif target_model == 'logistic':
        result = fit_logistic_mechanistic(x, y)

    if result and "error" not in result:
        print(f"Model: {result['type']}")
        print(f"Equation: {result['equation']}")
        print(f"R²: {result['r_squared']:.4f}")
        if 'aic' in result:
            print(f"AIC: {result['aic']:.2f}")
            print(f"BIC: {result['bic']:.2f}")
        if 'interpretation' in result:
            print(f"Interpretation: {result['interpretation']}")
    elif result:
        print("Error:", result['error'])
        sys.exit(1)
    
    # Visualization for single model
    if args.plot and result and "error" not in result:
        plot_results(x, y, [result], peaks)

if __name__ == "__main__":
    main()

