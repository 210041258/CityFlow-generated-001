import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -------------------------
# Load CSV data
# -------------------------
csv_file = "C:/Users/asdal/Downloads/CityFlow-master/tools/generator/vehicle_positions.csv"
df = pd.read_csv(csv_file)

# -------------------------
# Prepare figure
# -------------------------
fig, ax = plt.subplots(figsize=(10, 8))
scat = ax.scatter([], [], s=50, c='blue')  # initial empty scatter

ax.set_xlabel("X position")
ax.set_ylabel("Y position")
ax.set_title("CityFlow Traffic Simulation")

# Adjust axes limits based on data
ax.set_xlim(df['x'].min() - 5, df['x'].max() + 5)
ax.set_ylim(df['y'].min() - 5, df['y'].max() + 5)

# -------------------------
# Animation function
# -------------------------
steps = sorted(df['step'].unique())

def update(frame):
    step = steps[frame]
    step_data = df[df['step'] == step]
    scat.set_offsets(step_data[['x', 'y']].values)
    ax.set_title(f"CityFlow Traffic Simulation - Step {step}")
    return scat,

# -------------------------
# Run animation
# -------------------------
ani = animation.FuncAnimation(
    fig, update, frames=len(steps), interval=200, blit=True, repeat=False
)

plt.show()
# To save the animation as a video file, uncomment the following line:
# ani.save('traffic_simulation.mp4', writer='ffmpeg', fps=5)

# -------------------------
# End of script
# -------------------------
# Note: Ensure you have 'ffmpeg' installed for saving the animation as a video file.

    
