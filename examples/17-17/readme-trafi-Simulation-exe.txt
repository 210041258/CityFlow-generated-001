# Basic usage (will generate both hard scenarios)
python TRAFI-Simulation.py --flow-file flow_17_17.json

# With custom parameters
python TRAFI-Simulation.py --flow-file flow_17_17.json --duration 600 --dt 0.5

# Specify output directory
python TRAFI-Simulation.py --flow-file flow_17_17.json --output-dir ./simulation_results
