# Basic usage (will generate both hard scenarios)
python TRAFI-Simulation.py --flow-file flow_30_30_turn.json

# With custom parameters
python TRAFI-Simulation.py --flow-file flow_30_30_turn.json --duration 600 --dt 0.5

# Specify output directory
python TRAFI-Simulation.py --flow-file flow_30_30_turn.json --output-dir ./simulation_results