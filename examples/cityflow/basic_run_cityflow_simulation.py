import cityflow
import json
import os

# Paths to your existing JSON files
folder_path = os.path.dirname(os.path.abspath(__file__))  # current folder
roadnet_file = os.path.join(folder_path, "roadnet_3_4.json")
flow_file = os.path.join(folder_path, "flow_3_4.json")
config_file = os.path.join(folder_path, "config.json")

# Create a CityFlow config JSON
config = {
    "interval": 1,
    "seed": 42,
    "roadnetFile": roadnet_file,
    "flowFile": flow_file,
    "threadNum": 1
}

# Save the config file
with open(config_file, "w") as f:
    json.dump(config, f, indent=4)

# Load CityFlow simulation engine
engine = cityflow.Engine(config_file, thread_num=1)

# Run simulation for 100 steps
for step in range(100):
    engine.next_step()
    if step % 10 == 0:
        print(f"Step {step} completed")

print("Simulation finished successfully!")
