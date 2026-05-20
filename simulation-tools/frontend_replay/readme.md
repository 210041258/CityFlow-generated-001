# CityFlow Replay Visualizer & Traffic Dynamics Analysis

This folder contains two related tools:

1. **`python_simu.py`**: renders a CityFlow/SUMO-style `roadnet.json` + `replay.txt` into static figures and an animated visualization.
2. **`current-test/currnet.PY`**: analyzes vehicle-count time series extracted from `replay.txt` using growth statistics and model selection (AIC/BIC) with candidate models (linear/polynomial, exponential, logistic).

---

## Repository layout

- `python_simu.py`  
  Visualizes traffic in an intersection grid using `roadnet.json` and frame-by-frame vehicle + traffic light states from `replay.txt`.

- `roadnet.json`  
  Network geometry: intersection nodes and road edges with lane counts and lane widths.

- `replay.txt`  
  Simulation replay frames. Each line encodes:
  - vehicles (position/orientation + size + status)
  - traffic light states per road

- `current-test/`  
  Analysis pipeline output + the analysis script:
  - `currnet.PY` (analysis)
  - `output/` (generated text reports)

- `current-test/output/*.txt`
  - `growth.txt`: growth statistics
  - `dynamic.txt`: dynamics statistics (growth pulses, congestion peaks, trend prediction)
  - `compare.txt`: AIC/BIC model comparison and best-fit equation
  - `degree.txt`, `exponential.txt`, `logistic.txt`: per-model fit details

---

## 1) Visualization: `python_simu.py`

### What it produces
- A **static snapshot** for frame 0:
  - `frame_000.png`
  - `frame_000.pdf`
- An **interactive animation** of vehicles and traffic lights (optional MP4 saving).

### Command
From this folder:

```bash
python python_simu.py --roadnet roadnet.json --replay replay.txt --output frame
```

### Optional arguments
- `--save` : save animation as `traffic_simulation.mp4`
- `--interval <ms>` : animation frame interval (default: 50)

Example:

```bash
python python_simu.py --roadnet roadnet.json --replay replay.txt --output frame --save
```

### Notes on input formats
- `roadnet.json` is expected to contain `static.nodes` and `static.edges` with:
  - node ids (including `intersection_1_1` as the central intersection)
  - edge lane counts `nLane` and lane widths `laneWidths`
- `replay.txt` is expected to have one frame per line in the format:
  - `vehicles_part ; traffic_lights_part`
  - vehicles are comma-separated entries, each entry has 7 tokens:
    `x y angle name status length width`

---

## 2) Traffic dynamics analysis: `current-test/currnet.PY`

This script extracts **vehicle counts per frame** from `replay.txt` (streaming) and then:
- computes growth statistics
- performs congestion peak detection
- runs rolling dynamics + trend prediction (optional)
- fits candidate growth models and compares them using **AIC/BIC** (optional)

### Requirements
- `numpy` is required.
- `scipy` is recommended for exponential + logistic fits and congestion peak detection.
- `matplotlib` is only needed if you use `--plot`.

### Basic vehicle growth stats
```bash
python current-test/currnet.PY replay.txt --growth
```

### Model comparison (AIC/BIC)
```bash
python current-test/currnet.PY replay.txt --compare
```

### Traffic dynamics (rolling window + congestion)
```bash
python current-test/currnet.PY replay.txt --dynamics --window 10 --iqr-factor 1.5
```

### Model fitting + plots (optional)
```bash
python current-test/currnet.PY replay.txt --compare --plot
```

---

## Example outputs (already present under `current-test/output/`)

From the current run reports:

- **Number of frames**: 980
- **Average vehicles/frame**: 266.12
- **Min / Max**: 12 (frame 0) / 311 (frame 885)

### Best model (from `compare.txt`)
Models are compared by **AIC/BIC** (lower is better). For this dataset, **Logistic** is the best fit:

- **Logistic**
  - `y = 302.2328 / (1 + e^(-0.0143 * (x - 101.5374)))`
  - `R² = 0.9861`
  - `AIC = 4043.81`

---

## Generated report files

You can treat the following text files as the final analysis artifacts:
- `current-test/output/growth.txt`
- `current-test/output/dynamic.txt`
- `current-test/output/compare.txt`
- `current-test/output/degree.txt`
- `current-test/output/exponential.txt`
- `current-test/output/logistic.txt`

---

## Reproducibility checklist

1. Put your `roadnet.json` and `replay.txt` in this folder.
2. Run:
   - Visualization: `python python_simu.py ...`
   - Analysis: `python current-test/currnet.PY replay.txt --compare --dynamics`
3. Confirm the output files in `current-test/output/` match your run.

---

## Known warnings

- A runtime warning may appear during logistic fitting (overflow in `exp`) depending on SciPy/curve-fit numerical stability. The comparison output remains usable.

