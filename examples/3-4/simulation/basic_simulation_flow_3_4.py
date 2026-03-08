import json
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Load flow JSON
# -------------------------
with open("C:/Users/asdal/Downloads/CityFlow-master/tools/generator/flow_3_4.json") as f:
    flows = json.load(f)

# -------------------------
# Simulation parameters
# -------------------------
dt = 1.0            # 1 second per step
total_steps = 50    # simulate 50 steps
roads_length = 100  # assume each road segment = 100 meters

# -------------------------
# Prepare vehicles
# -------------------------
vehicles = []
for idx, flow in enumerate(flows):
    vehicle = flow["vehicle"]
    route = flow["route"]
    vehicles.append({
        "id": idx,
        "route": route,
        "pos": 0,               # start at beginning of first road
        "road_idx": 0,
        "speed": 0.0,
        "max_speed": vehicle["maxSpeed"],
        "acc": vehicle["usualPosAcc"],
        "length": vehicle["length"],
        "positions": []         # store trajectory for plotting
    })

# -------------------------
# Simulation loop
# -------------------------
for step in range(total_steps):
    for v in vehicles:
        # Compute distance to next vehicle on same road
        road_idx = v["road_idx"]
        same_road = [veh for veh in vehicles if veh["road_idx"] == road_idx and veh["pos"] > v["pos"]]
        if same_road:
            front = min([veh["pos"] for veh in same_road])
            v_gap = max(0.0, front - v["pos"] - v["length"])
            v_target = min(v["max_speed"], v_gap / dt)
        else:
            v_target = v["max_speed"]

        # Update speed and position
        v["speed"] = min(v["speed"] + v["acc"] * dt, v_target)
        v["pos"] += v["speed"] * dt

        # Move to next road if reached end
        if v["pos"] >= roads_length:
            if v["road_idx"] < len(v["route"]) - 1:
                v["road_idx"] += 1
                v["pos"] = 0
            else:
                v["speed"] = 0  # reached destination

        v["positions"].append((v["road_idx"], v["pos"]))

# -------------------------
# Plot trajectories
# -------------------------
plt.figure(figsize=(12,6))
for v in vehicles:
    traj = np.array(v["positions"])
    plt.plot(traj[:,1] + traj[:,0]*roads_length, label=f'Vehicle {v["id"]}')
plt.xlabel("Distance along route (m)")
plt.ylabel("Vehicle position (m)")
plt.title("Theoretical Vehicle Trajectories")
plt.legend()
plt.grid(True)
plt.show()
# -------------------------

# End of simulation.py

