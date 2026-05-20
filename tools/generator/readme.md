# Generator

`generate_grid_scenario.py` generates an **NxM grid road network** along with **traffic flow** definitions (and, optionally, a predefined traffic light plan).

---

## Quick Start

### 1) Minimal example (3x4)

```bash
python generate_grid_scenario.py 3 4
```

### 2) 3x4 grid with more straight lanes + traffic light plan

```bash
python generate_grid_scenario.py 3 4 --numStraightLanes 2 --tlPlan
```

### 3) Turning flows instead of straight flows

```bash
python generate_grid_scenario.py 3 4 --turn
```

---

## What gets generated

By default, the script writes a **road network** JSON and a **flow** JSON into the output directory.

- `--roadnetFile`: generated road network file (default: `roadnetFile` argument in the script; commonly `roadnet.json`)
- `--flowFile`: generated flow file (default: `flowFile` argument in the script; commonly `flow.json`)
- `--dir`: output directory (default: `./`)

(If you changed `--dir`, the output files will appear in that folder.)

You can check the repository for example outputs such as `roadnet.json` and `flow.json`.

---

## Other arguments

### Grid geometry
- `--rowDistance`: int, default `300` — distance between consecutive intersections along **East–West** roads (meters)
- `--columnDistance`: int, default `300` — distance between consecutive intersections along **South–North** roads (meters)
- `--intersectionWidth`: int, default `30` — intersection width (meters)

### Lane configuration
- `--numLeftLanes`: int, default `1`
- `--numStraightLanes`: int, default `1`
- `--numRightLanes`: int, default `1`

### Vehicle model
- `--laneMaxSpeed`: int/float, default `16.67` — lane max speed (m/s)
- `--vehLen`: float, default `5.0` — vehicle length (m)
- `--vehWidth`: float, default `2.0` — vehicle width (m)
- `--vehMaxPosAcc`: float, default `2.0`
- `--vehMaxNegAcc`: float, default `4.5`
- `--vehUsualPosAcc`: float, default `2.0`
- `--vehUsualNegAcc`: float, default `4.5`
- `--vehMinGap`: float, default `2.5` — minimum gap (m)
- `--vehMaxSpeed`: float, default `16.67`
- `--vehHeadwayTime`: float, default `1.5` — headway time (s)

### Output / behavior
- `--dir`: str, default `./` — output directory
- `--roadnetFile`: str — generated road network file name
- `--flowFile`: str — generated flow file name

### Scenario switches
- `--turn`: generate **turning** flows instead of straight flows
- `--tlPlan`: generate a working predefined traffic signal plan instead of plans with default orders
- `--interval`: float, default `2.0` — time (seconds) between vehicles for each flow

---

## Notes
- The script is primarily parameterized by **grid size (N, M)** plus lane counts and (optionally) traffic light plan/turning behavior.
- If you generate for a new experiment, start from the minimal example and only adjust the parameters you need.

