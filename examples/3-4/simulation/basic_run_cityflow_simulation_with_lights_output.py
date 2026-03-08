import cityflow
import json
import csv
import os

# -------------------------
# Paths
# -------------------------
config_file = "C:/Users/asdal/Downloads/CityFlow-master/tools/generator/config.json"
output_csv = "C:/Users/asdal/Downloads/CityFlow-master/tools/generator/vehicle_positions_lights.csv"

# Ensure folder exists
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# -------------------------
# Load engine
# -------------------------
engine = cityflow.Engine(config_file, thread_num=1)

# -------------------------
# Simulation parameters
# -------------------------
total_steps = 100

# Get traffic light IDs
traffic_light_ids = engine.get_traffic_light_ids()

# -------------------------
# Prepare CSV
# -------------------------
with open(output_csv, mode='w', newline='') as csvfile:
    fieldnames = ["step", "vehicle_id", "x", "y", "speed"] + [f"tl_{tl}" for tl in traffic_light_ids]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # -------------------------
    # Run simulation
    # -------------------------
    for step in range(1, total_steps + 1):
        engine.next_step()
        vehicle_ids = engine.get_vehicle_ids()

        # Get traffic light states (phase index)
        tl_states = {f"tl_{tl}": engine.get_traffic_light_phase(tl) for tl in traffic_light_ids}

        for vid in vehicle_ids:
            pos = engine.get_vehicle_info(vid, "position")
            speed = engine.get_vehicle_info(vid, "speed")
            row = {"step": step, "vehicle_id": vid, "x": pos[0], "y": pos[1], "speed": speed}
            row.update(tl_states)
            writer.writerow(row)

        if step % 10 == 0:
            print(f"Step {step} - Total vehicles: {len(vehicle_ids)}")

print(f"\nSimulation finished! Data saved to {output_csv}")
