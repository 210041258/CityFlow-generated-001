import cityflow
import json
import csv
import os

# -------------------------
# Paths
# -------------------------
config_file = "C:/Users/asdal/Downloads/CityFlow-master/tools/generator/config.json"
output_csv = "C:/Users/asdal/Downloads/CityFlow-master/tools/generator/vehicle_positions.csv"

# Ensure the output folder exists
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# -------------------------
# Load CityFlow engine
# -------------------------
engine = cityflow.Engine(config_file, thread_num=1)

# -------------------------
# Simulation parameters
# -------------------------
total_steps = 100  # Total steps
print_interval = 10  # Print to console every N steps

# -------------------------
# Prepare CSV file
# -------------------------
with open(output_csv, mode='w', newline='') as csvfile:
    fieldnames = ["step", "vehicle_id", "x", "y", "speed"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # -------------------------
    # Run simulation
    # -------------------------
    for step in range(1, total_steps + 1):
        engine.next_step()
        vehicle_ids = engine.get_vehicle_ids()

        # Write each vehicle's info to CSV
        for vid in vehicle_ids:
            pos = engine.get_vehicle_info(vid, "position")
            speed = engine.get_vehicle_info(vid, "speed")
            writer.writerow({
                "step": step,
                "vehicle_id": vid,
                "x": pos[0],
                "y": pos[1],
                "speed": speed
            })

        # Print summary every 'print_interval' steps
        if step % print_interval == 0:
            print(f"Step {step} - Total vehicles: {len(vehicle_ids)}")

print(f"\nSimulation finished! Vehicle positions saved to:\n{output_csv}")
