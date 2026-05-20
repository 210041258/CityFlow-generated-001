#!/usr/bin/env python3
"""
Replay File Analysis Tool (Enhanced)

This script parses SUMO replay files and provides advanced analysis:
1. Growth analysis: vehicle counts per frame, statistics.
2. Model fitting: Linear, Polynomial, Exponential, and Logistic.
3. Traffic Dynamics: Growth, Congestion, and Trend Prediction.
4. Model comparison: Ranks models using AIC and BIC.

Usage:
    python main.py <replay_file> [--model TYPE] [--degree N] [--compare] [--dynamics]

Examples:
    python main.py replay.txt --model linear
    python main.py replay.txt --model exponential
    python main.py replay.txt --model logistic
    python main.py replay.txt --compare          # Compare all models
    python main.py replay.txt --dynamics         # Traffic dynamics analysis
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

# ----------------------------------------------------------------------
# Parsing function
# ----------------------------------------------------------------------
def parse_replay(filename):
    """Parses a SUMO replay file into a list of frames."""
    frames = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(';')
                if len(parts) != 2:
                    continue
                veh_str, tl_str = parts

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
    except FileNotFoundError:
        return None
    return frames

# ----------------------------------------------------------------------
# Helper: Calculate Statistics
# ----------------------------------------------------------------------
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

    # AIC and BIC for Least Squares (assuming Gaussian errors)
    # AIC = n * ln(RSS/n) + 2k
    # BIC = n * ln(RSS/n) + k * ln(n)
    
    if rss <= 0 or n <= k:
        aic = float('inf')
        bic = float('inf')
    else:
        aic = n * np.log(rss / n) + 2 * k
        bic = n * np.log(rss / n) + k * np.log(n)
        
    return r2, aic, bic

# ----------------------------------------------------------------------
# Model Fitting Functions
# ----------------------------------------------------------------------
def fit_polynomial_model(x, y, degree):
    """Fits y = a0 + a1*x + ... + ad*x^d"""
    if degree < 1: degree = 1
    coefs = np.polyfit(x, y, degree)
    p = np.poly1d(coefs)
    y_pred = p(x)
    rss = np.sum((y - y_pred) ** 2)
    k = degree + 1
    
    r2, aic, bic = calculate_stats(len(x), k, rss, y, y_pred)
    
    # Format equation
    terms = []
    for i, c in enumerate(coefs):
        power = degree - i
        if abs(c) < 1e-4: continue
        if power == 0:
            terms.append(f"{c:.4f}")
        elif power == 1:
            terms.append(f"{c:.4f}*x")
        else:
            terms.append(f"{c:.4f}*x^{power}")
    eq = "y = " + " + ".join(terms) if terms else "y = 0"
    
    return {
        "type": f"Polynomial (deg {degree})",
        "equation": eq,
        "r_squared": r2,
        "aic": aic,
        "bic": bic,
        "params": coefs.tolist()
    }

def fit_exponential_model(x, y):
    """Fits y = a * exp(b * x)"""
    if not HAS_SCIPY:
        return {"error": "SciPy required for exponential fit."}
    
    # Filter non-positive y for numerical stability in initial guess, 
    # though curve_fit can handle raw data.
    # Model: y = a * e^(bx)
    func = lambda x, a, b: a * np.exp(b * x)
    
    # Initial guess: linear regression on log(y)
    # To avoid log(0), add a small epsilon or use only positive values
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

def fit_logistic_model(x, y):
    """Fits y = L / (1 + e^(-k * (x - x0)))"""
    if not HAS_SCIPY:
        return {"error": "SciPy required for logistic fit."}

    func = lambda x, L, k, x0: L / (1 + np.exp(-k * (x - x0)))
    
    # Initial Guesses
    L0 = max(y) * 1.1  # Slightly above max
    # Guess x0 as the point where y is half of L0
    # Find x where y is closest to L0/2
    half_max = L0 / 2
    idx = (np.abs(y - half_max)).argmin()
    x00 = x[idx]
    k0 = 1.0 # Arbitrary growth rate start
    
    try:
        popt, _ = curve_fit(func, x, y, p0=[L0, k0, x00], maxfev=10000)
        L, k, x0 = popt
        y_pred = func(x, L, k, x0)
        rss = np.sum((y - y_pred) ** 2)
        r2, aic, bic = calculate_stats(len(x), 3, rss, y, y_pred)
        
        eq = f"y = {L:.4f} / (1 + e^(-{k:.4f} * (x - {x0:.4f})))"
        return {
            "type": "Logistic",
            "equation": eq,
            "r_squared": r2,
            "aic": aic,
            "bic": bic,
            "params": [L, k, x0]
        }
    except Exception as e:
        return {"error": f"Logistic fit failed: {str(e)}"}

# ----------------------------------------------------------------------
# Analysis Functions
# ----------------------------------------------------------------------
def analyze_replay_growth(filename):
    """Basic growth statistics."""
    frames = parse_replay(filename)
    if frames is None or not frames:
        return {"error": "No frames found or file not found."}

    counts = [len(f['vehicles']) for f in frames]
    n = len(counts)
    avg = sum(counts) / n if n else 0
    return {
        "frame_count": n,
        "counts": counts,
        "average": avg,
        "min": min(counts),
        "max": max(counts)
    }

def run_comparison(x, y):
    """Runs all available models and compares them."""
    results = []
    
    # 1. Linear (Poly deg 1)
    res_lin = fit_polynomial_model(x, y, 1)
    if "error" not in res_lin: results.append(res_lin)
    
    # 2. Quadratic (Poly deg 2)
    res_quad = fit_polynomial_model(x, y, 2)
    if "error" not in res_quad: results.append(res_quad)
    
    # 3. Exponential
    if HAS_SCIPY:
        res_exp = fit_exponential_model(x, y)
        if "error" not in res_exp: results.append(res_exp)
        
        # 4. Logistic
        res_log = fit_logistic_model(x, y)
        if "error" not in res_log: results.append(res_log)
            
    # Sort by AIC (lower is better)
    results.sort(key=lambda k: k['aic'])
    
    return results

# ----------------------------------------------------------------------
# Traffic Dynamics Analysis Functions
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
    # Calculate difference: counts[i+1] - counts[i]
    # Prepend 0 so the array length matches the original counts
    growth_rates = np.diff(counts, prepend=0)
    
    # Relative Growth: (change / previous_count)
    # Initialize with zeros
    relative_growth = np.zeros(n_frames)
    # Create mask where previous count > 0 to avoid division by zero
    prev_counts = counts[:-1]
    valid_mask = prev_counts > 0
    
    # Align indices: relative change at frame i+1 depends on count at i
    relative_growth[1:][valid_mask] = growth_rates[1:][valid_mask] / prev_counts[valid_mask]

    # Significant Growth Detection (using IQR instead of Mean/Std)
    # We only look at positive growth to detect increases
    positive_growth = growth_rates[growth_rates > 0]
    
    if len(positive_growth) > 0:
        q75, q25 = np.percentile(positive_growth, [75, 25])
        iqr = q75 - q25
        # Threshold = Q3 + factor * IQR
        threshold = q75 + (iqr_factor * iqr)
        significant_growth_frames = np.where(growth_rates > threshold)[0].tolist()
    else:
        threshold = 0
        significant_growth_frames = []

    # Cumulative Growth (Total change from start)
    cumulative_growth = np.cumsum(growth_rates)

    # ---- 2. Congestion Detection (Improved Peaks) ----
    # Dynamic Height: Peak must be above baseline
    height_th = np.mean(counts) + np.std(counts)
    
    # Prominence: Peak must be significantly higher than immediate neighbors
    # This filters out single-tick noise in high-traffic areas
    if peak_prominence is None:
        peak_prominence = np.std(counts)
        
    peaks, _ = find_peaks(
        counts, 
        height=height_th, 
        prominence=peak_prominence,
        distance=window_size # Ensure peaks are at least 'window_size' apart
    )

    # ---- 3. Trend Smoothing (Rolling Average) ----
    # Use 'same' mode to align with original index (padded with zeros at edges)
    kernel = np.ones(window_size) / window_size
    rolling_avg = np.convolve(counts, kernel, mode='same')
    
    # Boundary correction: The 'same' mode pads with zeros, which dips the average at the start/end.
    # We can optionally normalize the edges or leave as is depending on preference.
    # For simplicity, we leave it, but note that the first 'window_size' points are slightly underestimated.

    # ---- 4. Trend Prediction (Rolling Linear Regression) ----
    # Optimized: Using numpy.polyfit instead of sklearn for speed and simplicity
    predicted_trend = np.full(n_frames, np.nan) # Initialize with NaNs
    
    # We need at least 2 points to fit a line
    if window_size >= 2:
        for i in range(n_frames - window_size + 1):
            # Extract window
            window_counts = counts[i : i + window_size]
            window_x = np.arange(window_size)
            
            # Fit degree 1 polynomial (Linear Regression)
            # coefs = [slope, intercept]
            coefs = np.polyfit(window_x, window_counts, 1)
            slope, intercept = coefs[0], coefs[1]
            
            # Predict the NEXT value (at index window_size)
            next_val = slope * window_size + intercept
            
            # Store prediction at the end of the window
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
# Main CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Analyze vehicle growth in SUMO replay files.")
    parser.add_argument("filename", help="Path to the replay file")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model", choices=['linear', 'exponential', 'logistic'], 
                       help="Select a specific model to fit")
    group.add_argument("--compare", action="store_true", 
                       help="Compare all available models (Linear, Quadratic, Exponential, Logistic)")
    group.add_argument("--dynamics", action="store_true", 
                       help="Run traffic dynamics analysis (Growth, Congestion, Trend)")
    
    parser.add_argument("--degree", type=int, default=2, 
                        help="Polynomial degree if 'linear' (interpreted as polynomial) is chosen (default: 2)")
    parser.add_argument("--growth", action="store_true", help="Show basic growth stats only")
    parser.add_argument("--window", type=int, default=10, 
                        help="Window size for traffic dynamics analysis (default: 10)")
    parser.add_argument("--iqr-factor", type=float, default=1.5, 
                        help="IQR factor for significant growth detection (default: 1.5)")
    
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

    # 2. Prepare Data
    frames = parse_replay(args.filename)
    if frames is None:
        print("Error: File not found.")
        sys.exit(1)
    if not frames:
        print("Error: No data in file.")
        sys.exit(1)
        
    x = np.arange(len(frames))
    y = np.array([len(f['vehicles']) for f in frames])

    # 3. Traffic Dynamics Analysis Mode
    if args.dynamics:
        run_traffic_dynamics_analysis(x, y, window_size=args.window, iqr_factor=args.iqr_factor)
        sys.exit(0)

    # 4. Model Comparison Mode
    if args.compare:
        print(f"Model Comparison for '{args.filename}'")
        print("=" * 80)
        results = run_comparison(x, y)
        
        if not results:
            print("No models could be fitted.")
            sys.exit(1)
            
        print(f"{'Model':<15} | {'R²':<8} | {'AIC':<10} | {'BIC':<10}")
        print("-" * 60)
        for r in results:
            print(f"{r['type']:<15} | {r['r_squared']:.4f}   | {r['aic']:.2f}     | {r['bic']:.2f}")
        
        print("\nBest Fit (Lowest AIC):")
        best = results[0]
        print(f"Type: {best['type']}")
        print(f"Equation: {best['equation']}")
        print(f"R²: {best['r_squared']:.4f}")
        sys.exit(0)

    # 4. Single Model Mode
    # Note: User selects 'linear', but we use the --degree argument to allow polynomial fits
    target_model = args.model
    if not target_model:
        # Default behavior if no model specified (fallback to linear/polynomial)
        target_model = 'linear' 

    result = None
    if target_model == 'linear':
        # Using --degree to specify polynomial degree for 'linear' request
        # If user wants strictly line, they should have passed --degree 1, but default is 2
        result = fit_polynomial_model(x, y, args.degree)
    elif target_model == 'exponential':
        result = fit_exponential_model(x, y)
    elif target_model == 'logistic':
        result = fit_logistic_model(x, y)

    if result and "error" not in result:
        print(f"Model: {result['type']}")
        print(f"Equation: {result['equation']}")
        print(f"R²: {result['r_squared']:.4f}")
        if 'aic' in result:
            print(f"AIC: {result['aic']:.2f}")
            print(f"BIC: {result['bic']:.2f}")
    elif result:
        print("Error:", result['error'])
        sys.exit(1)

if __name__ == "__main__":
    main()