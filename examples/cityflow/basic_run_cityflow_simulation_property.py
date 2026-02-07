import cityflow
import json

# -------------------------
# Load config
# -------------------------
config_file = "C:/Users/asdal/Downloads/CityFlow-master/tools/generator/config.json"

engine = cityflow.Engine(config_file, thread_num=1)

# -------------------------
# Simulation parameters
# -------------------------
total_steps = 100  # Total simulation steps
print_interval = 10  # Print every N steps

# -------------------------
# Run simulation
# -------------------------
for step in range(1, total_steps + 1):
    engine.next_step()

    # Print vehicle positions every 'print_interval' steps
    if step % print_interval == 0:
        vehicle_ids = engine.get_vehicle_ids()
        print(f"\nStep {step} - Total vehicles: {len(vehicle_ids)}")
        for vid in vehicle_ids:
            pos = engine.get_vehicle_info(vid, "position")  # x, y coordinates
            speed = engine.get_vehicle_info(vid, "speed")
            print(f"Vehicle {vid}: Position {pos}, Speed {speed:.2f}")

print("\nSimulation finished!")

# -------------------------
# Clean up
engine.close()
# -------------------------
# %% End of file %%

# C:/Users/asdal/AppData/Local/Microsoft/WindowsApps/python3.13.exe C:/Users/asdal/Downloads/CityFlow-master/tools/generator/run_simulation.py
# Step 10 - Total vehicles: 15
# Vehicle 0: Position [12.3, 34.5], Speed 5.40
# Vehicle 1: Position [5.0, 10.2], Speed 3.20
# ...
# Step 20 - Total vehicles: 18
# Vehicle 0: Position [15.6, 38.1], Speed 5.60
# Vehicle 1: Position [7.3, 12.5], Speed 3.   
