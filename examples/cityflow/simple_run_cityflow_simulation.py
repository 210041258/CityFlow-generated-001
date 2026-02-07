import cityflow
import json

# Load config
config_file = "C:/Users/asdal/Downloads/CityFlow-master/tools/generator/config.json"

engine = cityflow.Engine(config_file, thread_num=1)

# Run simulation for 100 steps
for step in range(100):
    engine.next_step()

print("Simulation finished!")
