# Output Analysis Report (current-test/output)

This folder contains the results of running the traffic replay analysis pipeline on `../replay.txt` (vehicle-count per frame) using `../currnet.PY`.

## Contents
- `compare.txt` — Model comparison using AIC/BIC (linear/poly deg1, poly deg2, exponential, logistic)
- `degree.txt` — Polynomial (deg 2) fit details
- `dynamic.txt` — Traffic dynamics analysis (growth statistics, congestion peaks, trend prediction)
- `exponential.txt` — Exponential model fit details
- `growth.txt` — Vehicle growth summary over frames
- `logistic.txt` — Logistic model fit details

## Data summary (from `growth.txt`)
- Number of frames: **980**
- Average vehicles per frame: **266.12**
- Maximum vehicles: **311** (frame **885**)
- Minimum vehicles: **12** (frame **0**)
- Total growth (last - first): **285**

## Traffic dynamics (from `dynamic.txt`)
- Window size: **10**
- IQR factor: **1.5**

### Growth statistics
- Mean growth rate: **0.30**
- Max growth rate: **12.00**
- Min growth rate: **-6.00**
- Significant growth frames (detected by IQR threshold):
  - `[0, 5, 10, 15, 20, 25, 30, 35, 60, 90, 95, 100, 105, 110, 115, 120, 125, 165, 170, 230, 235]`

### Congestion detection
- Peak frames (congestion): `[]`
- Number of peaks: `0`

### Trend prediction
- Average predicted value: **268.73**
- First 5 predictions: `[28.0, 33.6, 37.6, 40.0, 40.8]`

## Model fitting summary (from `compare.txt`)
Models are compared by **AIC** (lower is better) and **BIC**.

| Model | R² | AIC | BIC |
|---|---:|---:|---:|
| Logistic | 0.9861 | 4043.81 | 4058.47 |
| Polynomial (deg 2) | 0.8892 | 6080.37 | 6095.03 |
| Polynomial (deg 1) | 0.5376 | 7478.30 | 7488.08 |
| Exponential | 0.4736 | 7605.22 | 7614.99 |

### Best fit (lowest AIC)
- **Logistic**
- Equation:
  - `y = 302.2328 / (1 + e^(-0.0143 * (x - 101.5374)))`
- R²: **0.9861**

## Per-model details
- `degree.txt` — Polynomial degree-2 equation and metrics (R², AIC/BIC)
- `exponential.txt` — Exponential equation and metrics (R², AIC/BIC)
- `logistic.txt` — Logistic equation and metrics (R², AIC/BIC)

## How to reproduce (from `currnet.PY`)
Run the analysis script against the replay file:

```bash
python currnet.PY ../replay.txt --compare --dynamics
```

Useful options inside `currnet.PY`:
- `--growth` : only basic vehicle growth statistics
- `--model {linear,exponential,logistic}` : fit a specific model
- `--window <int>` : window size for dynamics analysis (default 10)
- `--iqr-factor <float>` : IQR multiplier for significant growth frames (default 1.5)
- `--plot` : show matplotlib plots (requires matplotlib)

## Notes
- `logistic.txt` / `compare.txt` may include a runtime warning related to exponential overflow in the logistic curve fitting implementation.

---

## Mathematical Methodology & Model Selection

### Goal
Select a functional form that explains how the vehicle count evolves over frames, while avoiding overfitting.

This report compares candidate models using **information criteria** (AIC/BIC), which trade off:
- **fit quality** (how close the model is to the data)
- **model complexity** (how many parameters the model uses)


### Models considered
1. **Linear** (constant growth)
   - Form: \( y = m x + c \)
   - Reason for rejection (in traffic-density terms): implies unbounded/infinite growth. Traffic density is bounded by road capacity, so perpetual linear growth is physically implausible.

2. **Exponential** (accelerating growth)
   - Form: \( y = a e^{b x} \)
   - Reason for rejection: exponential growth lacks a saturation mechanism; it implies the system can accelerate indefinitely. Congestion acts as a “drag” that prevents runaway acceleration.

3. **Polynomial (degree 2)** (parabolic trajectory)
   - Form: \( y = a x^2 + b x + c \)
   - Reason for rejection: a parabola either opens downward (eventual decrease) or opens upward (infinite acceleration); neither matches a steady-state/bounded traffic fill.

4. **Logistic** (bounded S-curve growth)
   - Form: \( y = \frac{L}{1 + e^{-k(x-x_0)}} \)
   - Why it fits traffic: logistic curves provide a natural saturation (“carrying capacity”) \(L\) and an inflection point \(x_0\) where the growth transitions from fast to slow.

### Statistical justification (AIC/BIC)
The model selection criterion uses **AIC** and **BIC**, which penalize parameter count while rewarding fit quality.

- **AIC**: \(\mathrm{AIC} = n\ln(\widehat{\mathrm{RSS}}/n) + 2k\)
- **BIC**: \(\mathrm{BIC} = n\ln(\widehat{\mathrm{RSS}}/n) + k\ln(n)\)

Where:
- \(n\): number of observations (frames)
- \(k\): number of fitted parameters
- \(\widehat{\mathrm{RSS}}\): residual sum of squares for the fitted model

Interpretation:
- Lower AIC/BIC is better.
- **AIC meaning**: an (approximate) estimator of out-of-sample prediction error; it balances fit quality vs. parameter count (overfitting control).
- **BIC meaning**: a Bayesian-flavored score with a stronger complexity penalty; with large \(n\) (here \(n=980\)), BIC increasingly prefers simpler models unless the fit improvement is substantial.
- Large differences \(\Delta\mathrm{AIC}\) (or \(\Delta\mathrm{BIC}\)) indicate that the runner-up model is strongly disfavored.


Using this dataset:
- Best model (lowest AIC): **Logistic** with **AIC = 4043.81**
- Runner-up: **Polynomial (deg 2)** with **AIC = 6080.37**
- \(\Delta\mathrm{AIC} \approx 2036\), which is far above common “decisive evidence” thresholds (e.g., \(\Delta\mathrm{AIC} > 10\)).

### Dynamics & signal-processing interpretation
This run also performs a rolling / windowed dynamics analysis:
- **Saturation analysis**: the logistic parameter **\(L \approx 302.23\)** represents the theoretical saturation limit. The observed maximum (**311 vehicles**) can be interpreted as a transient overshoot or measurement noise around the equilibrium region.
- **Growth pulses**: the “significant growth frames” list shows repeated bursts separated by a roughly consistent interval, suggesting the input process may contain periodic/queued batch effects on top of the slower logistic fill.
- **Congestion null result**: the IQR-based growth anomaly detector found no congestion peaks. Near saturation, variance can decrease (the logistic “pinch”), making peak-based detectors less sensitive. In this context, “congestion” is treated statistically as deviation from the model trend rather than as simply “high vehicle count.”

### How to reproduce the model selection logic
Run the comparison mode (prints AIC/BIC and equation for each supported model):

```bash
python currnet.PY ../replay.txt --compare --dynamics
```

This calls:
- Logistic fitting (`fit_logistic_mechanistic`) using SciPy `curve_fit`
- Polynomial fitting (`fit_polynomial_empirical`) using NumPy `polyfit`
- Exponential fitting (`fit_exponential_model`) using SciPy `curve_fit`
- Information criteria computation (`calculate_information_criteria`) for AIC/BIC ranking


