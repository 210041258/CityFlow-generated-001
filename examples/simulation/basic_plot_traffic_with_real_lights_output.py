import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
import numpy as np

# -------------------------
# Paths
# -------------------------
csv_file = "C:/Users/asdal/Downloads/CityFlow-master/tools/generator/vehicle_positions_lights.csv"
roadnet_file = "C:/Users/asdal/Downloads/CityFlow-master/tools/generator/roadnet_3_4.json"

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(csv_file)

# Load road network
with open(roadnet_file, "r") as f:
    roadnet = json.load(f)

# Road segments
lines = []
for road in roadnet['roads'].values():
    lane = road['lanes'][0]
    start = lane['start']
    end = lane['end']
    lines.append((start, end))
lines = np.array(lines)

# Traffic light positions
junctions = roadnet.get('trafficLights', {})
light_positions = []
light_ids = []
for tl_id, tl in junctions.items():
    light_positions.append(tl['pos'])
    light_ids.append(f"tl_{tl_id}")
light_positions = np.array(light_positions)

# -------------------------
# Prepare figure
# -------------------------
fig, ax = plt.subplots(figsize=(10, 8))

# Plot roads
for start, end in lines:
    ax.plot([start[0], end[0]], [start[1], end[1]], color='gray', linewidth=2)

# Vehicles scatter
scat = ax.scatter([], [], s=50, c=[], cmap='jet', vmin=0, vmax=df['speed'].max())

# Traffic lights scatter
lights_scat = ax.scatter([], [], s=150, c=[], marker='s')

ax.set_xlabel("X position")
ax.set_ylabel("Y position")
ax.set_title("CityFlow Traffic Simulation with Real Lights")

# Axes limits
all_coords = lines.reshape(-1, 2)
ax.set_xlim(all_coords[:,0].min() - 5, all_coords[:,0].max() + 5)
ax.set_ylim(all_coords[:,1].min() - 5, all_coords[:,1].max() + 5)

# Steps
steps = sorted(df['step'].unique())

# -------------------------
# Animation function
# -------------------------
def update(frame):
    step = steps[frame]
    step_data = df[df['step'] == step]
    positions = step_data[['x', 'y']].values
    speeds = step_data['speed'].values
    scat.set_offsets(positions)
    scat.set_array(speeds)

    # Update traffic lights
    tl_colors = []
    for tl in light_ids:
        phase = step_data[tl].iloc[0]  # get phase index
        # Simplified coloring: 0 = green, others = red
        tl_colors.append('green' if phase == 0 else 'red')
    if len(light_positions) > 0:
        lights_scat.set_offsets(light_positions)
        lights_scat.set_color(tl_colors)

    ax.set_title(f"CityFlow Traffic Simulation - Step {step}")
    return scat, lights_scat

# -------------------------
# Animate
# -------------------------
ani = animation.FuncAnimation(
    fig, update, frames=len(steps), interval=200, blit=True, repeat=False
)

# Colorbar for speed
cbar = plt.colorbar(scat, ax=ax)
cbar.set_label('Vehicle Speed')

plt.show()
# Save animation as GIF
#ani.save("traffic_simulation_with_lights.gif", writer='pillow', fps=5)