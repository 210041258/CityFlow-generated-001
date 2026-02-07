import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
import numpy as np

# -------------------------
# Paths
# -------------------------
csv_file = "C:/Users/asdal/Downloads/CityFlow-master/tools/generator/vehicle_positions.csv"
roadnet_file = "C:/Users/asdal/Downloads/CityFlow-master/tools/generator/roadnet_3_4.json"

# -------------------------
# Load CSV data
# -------------------------
df = pd.read_csv(csv_file)

# -------------------------
# Load road network
# -------------------------
with open(roadnet_file, "r") as f:
    roadnet = json.load(f)

# Extract road segments for plotting
lines = []
for road in roadnet['roads'].values():
    lane = road['lanes'][0]  # Take first lane
    start = lane['start']
    end = lane['end']
    lines.append((start, end))

lines = np.array(lines)

# Extract traffic light positions (junctions)
junctions = roadnet.get('trafficLights', {})  # dictionary
light_positions = []
for tl_id, tl in junctions.items():
    light_positions.append(tl['pos'])
light_positions = np.array(light_positions)

# -------------------------
# Prepare figure
# -------------------------
fig, ax = plt.subplots(figsize=(10, 8))

# Plot road network
for start, end in lines:
    ax.plot([start[0], end[0]], [start[1], end[1]], color='gray', linewidth=2)

# Scatter plot for vehicles
scat = ax.scatter([], [], s=50, c=[], cmap='jet', vmin=0, vmax=df['speed'].max())

# Traffic lights scatter
lights_scat = ax.scatter([], [], s=150, c=[], marker='s')

ax.set_xlabel("X position")
ax.set_ylabel("Y position")
ax.set_title("CityFlow Traffic Simulation with Lights")

# Adjust axes limits
all_coords = lines.reshape(-1, 2)
ax.set_xlim(all_coords[:,0].min() - 5, all_coords[:,0].max() + 5)
ax.set_ylim(all_coords[:,1].min() - 5, all_coords[:,1].max() + 5)

# -------------------------
# Animation function
# -------------------------
steps = sorted(df['step'].unique())

# Simulate traffic light signals (toggle red/green every 10 steps)
def get_light_colors(step):
    if len(light_positions) == 0:
        return []
    colors = []
    for i in range(len(light_positions)):
        if (step // 10) % 2 == i % 2:
            colors.append('green')
        else:
            colors.append('red')
    return colors

def update(frame):
    step = steps[frame]
    step_data = df[df['step'] == step]
    positions = step_data[['x', 'y']].values
    speeds = step_data['speed'].values
    scat.set_offsets(positions)
    scat.set_array(speeds)

    # Update traffic lights
    if len(light_positions) > 0:
        lights_scat.set_offsets(light_positions)
        lights_scat.set_color(get_light_colors(step))

    ax.set_title(f"CityFlow Traffic Simulation - Step {step}")
    return scat, lights_scat

# -------------------------
# Run animation
# -------------------------
ani = animation.FuncAnimation(
    fig, update, frames=len(steps), interval=200, blit=True, repeat=False
)

# Add colorbar for vehicle speed
cbar = plt.colorbar(scat, ax=ax)
cbar.set_label('Vehicle Speed')

plt.show()

# To save the animation, uncomment the following line:
# ani.save('traffic_simulation_with_lights.mp4', writer='ffmpeg', fps=5)
#import pandas as pd
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
#import json
#import numpy as np
# -------------------------
# Paths
