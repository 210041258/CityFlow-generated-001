import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.lines import Line2D

# Optional dependencies – graceful fallback
try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# ----------------------------------------------------------------------
# 1. Parse replay file (vehicles + traffic lights)
# ----------------------------------------------------------------------
def parse_replay(filename):
    frames = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(';')
            if len(parts) != 2:
                continue
            veh_str, tl_str = parts

            vehicles = []
            for entry in veh_str.split(','):
                if not entry:
                    continue
                tokens = entry.strip().split()
                if len(tokens) != 7:
                    continue
                x = float(tokens[0])
                y = float(tokens[1])
                angle = float(tokens[2])
                name = tokens[3]
                status = int(tokens[4])
                length = float(tokens[5])
                width = float(tokens[6])
                vehicles.append({
                    'x': x, 'y': y, 'angle': angle,
                    'name': name, 'status': status,
                    'length': length, 'width': width
                })

            traffic_lights = {}
            for entry in tl_str.split(','):
                if not entry:
                    continue
                tokens = entry.strip().split()
                if len(tokens) != 4:
                    continue
                road_name = tokens[0]
                colors = tokens[1:4]
                traffic_lights[road_name] = colors

            frames.append({'vehicles': vehicles, 'traffic_lights': traffic_lights})
    return frames

# ----------------------------------------------------------------------
# 2. Parse road network from JSON
# ----------------------------------------------------------------------
def parse_roadnet(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    nodes = {}
    for node in data['static']['nodes']:
        nid = node['id']
        outline = node['outline']
        outline_pts = [(outline[i], outline[i+1]) for i in range(0, len(outline), 2)]
        nodes[nid] = {
            'point': node['point'],
            'virtual': node['virtual'],
            'outline': outline_pts,
            'width': node.get('width', None)
        }

    edges = {}
    for edge in data['static']['edges']:
        eid = edge['id']
        edges[eid] = {
            'from': edge['from'],
            'to': edge['to'],
            'points': [tuple(p) for p in edge['points']],
            'nLane': edge['nLane'],
            'laneWidths': edge['laneWidths'],
            'total_width': sum(edge['laneWidths'])
        }

    return nodes, edges

# ----------------------------------------------------------------------
# 3. Generic geometry helpers (no hardcoding)
# ----------------------------------------------------------------------
def get_intersection_stop_points(intersection_outline, center_point):
    """
    For an axis-aligned intersection (like an octagon), find the four points
    on the outline where the cardinal direction roads meet.
    Returns dict: {'east': (x,y), 'west': (x,y), 'north': (x,y), 'south': (x,y)}
    """
    xs = [p[0] for p in intersection_outline]
    ys = [p[1] for p in intersection_outline]
    return {
        'east':  (max(xs), center_point[1]),
        'west':  (min(xs), center_point[1]),
        'north': (center_point[0], max(ys)),
        'south': (center_point[0], min(ys))
    }

def get_boundary_attach_point(node):
    """Generic: use the outline edge closest to the road direction."""
    # For simplicity, we still rely on the fact that boundary nodes are small
    # rectangles placed at ±300. We take the midpoint of the edge facing inward.
    pt = node['point']
    x, y = pt
    outline = node['outline']
    if x == -300:   # left boundary – attach at right edge of rectangle
        return (max(p[0] for p in outline), y)
    elif x == 300:  # right boundary – attach at left edge
        return (min(p[0] for p in outline), y)
    elif y == -300: # bottom boundary – attach at top edge
        return (x, max(p[1] for p in outline))
    elif y == 300:  # top boundary – attach at bottom edge
        return (x, min(p[1] for p in outline))
    else:
        return pt

def get_direction_vector(from_point, to_point):
    """Return cardinal direction ('east','west','north','south') of to_point relative to from_point."""
    dx = to_point[0] - from_point[0]
    dy = to_point[1] - from_point[1]
    if abs(dx) > abs(dy):
        return 'east' if dx > 0 else 'west'
    else:
        return 'north' if dy > 0 else 'south'

# ----------------------------------------------------------------------
# 4. Pre‑compute all road geometry (store data, not patches)
# ----------------------------------------------------------------------
def prepare_road_network(nodes, edges):
    road_fills_data = []
    lane_boundaries = []
    intersection_polys_data = []
    boundary_polys_data = []
    tl_positions = {}          # key: exit edge ID, value: list of (lane_idx, x, y)

    # Central intersection
    center_node = nodes['intersection_1_1']
    center_point = center_node['point']
    stop_points = get_intersection_stop_points(center_node['outline'], center_point)

    for eid, e in edges.items():
        from_id = e['from']
        to_id = e['to']
        from_node = nodes[from_id]
        to_node = nodes[to_id]
        pt_start = e['points'][0]
        pt_end = e['points'][1]

        total_width = e['total_width']
        half_width = total_width / 2.0
        lane_widths = e['laneWidths']

        # Direction and perpendicular
        dx = pt_end[0] - pt_start[0]
        dy = pt_end[1] - pt_start[1]
        length = np.hypot(dx, dy)
        if length == 0:
            continue
        dx_n, dy_n = dx / length, dy / length
        perp_x, perp_y = -dy_n, dx_n

        # ----- Adjust start / end to intersection stop line or boundary attach point -----
        if from_id == 'intersection_1_1':
            dir_from = get_direction_vector(center_point, to_node['point'])
            start = stop_points[dir_from]
        else:
            start = get_boundary_attach_point(from_node)

        if to_id == 'intersection_1_1':
            dir_to = get_direction_vector(center_point, from_node['point'])
            end = stop_points[dir_to]
        else:
            end = get_boundary_attach_point(to_node)

        # 1. Road surface polygon (data)
        sl = (start[0] + half_width * perp_x, start[1] + half_width * perp_y)
        sr = (start[0] - half_width * perp_x, start[1] - half_width * perp_y)
        er = (end[0]   - half_width * perp_x, end[1]   - half_width * perp_y)
        el = (end[0]   + half_width * perp_x, end[1]   + half_width * perp_y)
        road_fills_data.append([sl, sr, er, el])

        # 2. Lane boundaries (outer solid, inner dashed)
        offsets = [half_width]
        cum = 0.0
        for w in lane_widths[:-1]:
            cum += w
            offsets.append(half_width - cum)
        offsets.append(-half_width)

        for off in offsets:
            x1 = start[0] + off * perp_x
            y1 = start[1] + off * perp_y
            x2 = end[0]   + off * perp_x
            y2 = end[1]   + off * perp_y
            style = 'solid' if abs(off) == half_width else 'dashed'
            lw = 0.8 if abs(off) == half_width else 0.6
            lane_boundaries.append((x1, y1, x2, y2, style, lw, 'gray'))

        # 3. Traffic light positions (stop line, lane centres)
        if to_id == 'intersection_1_1':
            stop_point = end
            cum = 0.0
            positions = []
            for i, w in enumerate(lane_widths):
                centre_off = half_width - cum - w/2.0
                cx = stop_point[0] + centre_off * perp_x
                cy = stop_point[1] + centre_off * perp_y
                positions.append((i, cx, cy))
                cum += w

            # ----- Map entering edge → exit edge using common boundary node -----
            # Entering edge: boundary_node → intersection
            # Exit edge:     intersection → same boundary_node
            boundary_node_id = from_id if from_id != 'intersection_1_1' else to_id
            # Find the edge that starts at intersection_1_1 and ends at this boundary node
            exit_id = None
            for cand_id, cand in edges.items():
                if cand['from'] == 'intersection_1_1' and cand['to'] == boundary_node_id:
                    exit_id = cand_id
                    break
            if exit_id:
                tl_positions[exit_id] = positions

    # 4. Intersection polygons (central + virtual boundaries)
    for nid, node in nodes.items():
        outline = node['outline']
        if nid == 'intersection_1_1':
            intersection_polys_data.append(outline)
        else:
            boundary_polys_data.append(outline)

    return {
        'road_fills': road_fills_data,
        'lane_boundaries': lane_boundaries,
        'intersection_polys': intersection_polys_data,
        'boundary_polys': boundary_polys_data,
        'tl_positions': tl_positions,
        'stop_points': stop_points   # useful for debugging
    }

# ----------------------------------------------------------------------
# 5. Console summary (with tabulate if available)
# ----------------------------------------------------------------------
def print_road_summary(nodes, edges, tl_positions):
    print("\n" + "="*60)
    print("              ROAD NETWORK STATICS".center(60))
    print("="*60)

    # Intersections
    print("\n[ Intersections ]")
    inter_data = []
    for nid, node in nodes.items():
        inter_data.append([nid, "yes" if node['virtual'] else "no", node['point'], len(node['outline'])])
    if TABULATE_AVAILABLE:
        print(tabulate(inter_data, headers=["ID", "Virtual", "Center", "Vertices"], tablefmt="grid"))
    else:
        for row in inter_data:
            print(f"  {row[0]:<20} {row[1]:<6} {row[2]}  ({row[3]} pts)")

    # Road edges
    print("\n[ Road Edges ]")
    edge_data = []
    for eid, e in edges.items():
        edge_data.append([eid, e['from'], e['to'], e['nLane'], e['laneWidths'], f"{e['total_width']:.1f} m"])
    if TABULATE_AVAILABLE:
        print(tabulate(edge_data, headers=["Edge ID", "From", "To", "#Lanes", "Lane widths", "Total width"], tablefmt="grid"))
    else:
        for row in edge_data:
            print(f"  {row[0]:<15} {row[1]:<18} {row[2]:<18} {row[3]}  {row[4]}  {row[5]}")

    # Traffic light positions
    print("\n[ Traffic Light Positions (stop line, lane centres) ]")
    tl_data = []
    for exit_id, positions in tl_positions.items():
        for lane_idx, cx, cy in positions:
            tl_data.append([exit_id, lane_idx, f"({cx:.1f}, {cy:.1f})"])
    if TABULATE_AVAILABLE:
        print(tabulate(tl_data, headers=["Exit road", "Lane", "Stop line point"], tablefmt="grid"))
    else:
        for row in tl_data:
            print(f"  {row[0]:<15} lane {row[1]}: {row[2]}")
    print("\n" + "="*60 + "\n")

# ----------------------------------------------------------------------
# 6. Vehicle drawing – detailed, with stable colour and dynamic scaling
# ----------------------------------------------------------------------
def get_vehicle_color(name):
    """Deterministic stable color per vehicle."""
    hash_val = int(name.replace('vehicle', '')) if 'vehicle' in name else abs(hash(name)) % 1000
    return plt.cm.tab20(hash_val % 20)

def draw_detailed_car(ax, v, show_label=False):
    x, y, angle = v['x'], v['y'], v['angle']
    length, width = v['length'], v['width']
    name = v['name']
    color = get_vehicle_color(name)

    # Shadow (scaled with vehicle size)
    shadow_offset = max(length, width) * 0.08
    shadow = patches.Rectangle(
        (x - length/2 + shadow_offset, y - width/2 + shadow_offset),
        length, width,
        angle=np.degrees(angle), rotation_point='center',
        facecolor='black', alpha=0.2, zorder=9
    )
    ax.add_patch(shadow)

    # Car body
    rect = patches.Rectangle(
        (x - length/2, y - width/2), length, width,
        angle=np.degrees(angle), rotation_point='center',
        facecolor=color, edgecolor='black',
        linewidth=0.8, alpha=0.9, zorder=10
    )
    ax.add_patch(rect)

    # Lights – position relative to car orientation
    half_l = length / 2
    half_w = width / 2
    margin = min(length, width) * 0.1

    # Front-right, front-left, rear-right, rear-left
    corners_rel = [
        (half_l - margin, -half_w + margin),  # FR
        (half_l - margin, half_w - margin),   # FL
        (-half_l + margin, -half_w + margin), # RR
        (-half_l + margin, half_w - margin)   # RL
    ]

    # Rotation and translation
    s, c = np.sin(angle), np.cos(angle)
    corners_abs = []
    for px, py in corners_rel:
        xnew = x + px * c - py * s
        ynew = y + px * s + py * c
        corners_abs.append((xnew, ynew))

    light_radius = min(length, width) * 0.12
    # Headlights (front) – yellow
    ax.add_patch(patches.Circle(corners_abs[0], light_radius, facecolor='yellow', edgecolor='none', zorder=11))
    ax.add_patch(patches.Circle(corners_abs[1], light_radius, facecolor='yellow', edgecolor='none', zorder=11))
    # Taillights (rear) – red
    ax.add_patch(patches.Circle(corners_abs[2], light_radius, facecolor='red', edgecolor='none', zorder=11))
    ax.add_patch(patches.Circle(corners_abs[3], light_radius, facecolor='red', edgecolor='none', zorder=11))

    # Optional name label
    if show_label:
        ax.text(x, y + width/2 + 1.5, name, fontsize=6, ha='center', zorder=12)

# ----------------------------------------------------------------------
# 7. Common road drawing function (static background)
# ----------------------------------------------------------------------
def draw_road_network(ax, rn):
    """Draw all static road elements – used by both static plot and animation background."""
    for pts in rn['road_fills']:
        ax.add_patch(patches.Polygon(pts, closed=True, facecolor='lightgray', edgecolor='none', zorder=1))
    for x1, y1, x2, y2, style, lw, color in rn['lane_boundaries']:
        ax.plot([x1, x2], [y1, y2], linestyle=style, linewidth=lw, color=color, zorder=2)
    for pts in rn['intersection_polys']:
        ax.add_patch(patches.Polygon(pts, closed=True, facecolor='lightgray', edgecolor='black', linewidth=1.0, zorder=5))
    for pts in rn['boundary_polys']:
        ax.add_patch(patches.Polygon(pts, closed=True, facecolor='white', edgecolor='black', linewidth=0.8, zorder=5))

# ----------------------------------------------------------------------
# 8. Static publication frame (first frame)
# ----------------------------------------------------------------------
def plot_static_frame(frame, road_network, output_prefix="frame"):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(-320, 320)
    ax.set_ylim(-320, 320)
    ax.set_aspect('equal')
    ax.set_title("Intersection snapshot – Frame 0", fontsize=16, weight='bold')

    draw_road_network(ax, road_network)

    # Vehicles with labels
    for v in frame['vehicles']:
        draw_detailed_car(ax, v, show_label=True)

    # Traffic lights
    tl_data = frame['traffic_lights']
    positions = road_network['tl_positions']
    for road_id, colors in tl_data.items():
        if road_id not in positions:
            continue
        for lane_idx, cx, cy in positions[road_id]:
            col = colors[lane_idx]
            facecol = {'r': 'red', 'g': 'green', 'y': 'yellow'}.get(col, 'gray')
            ax.add_patch(patches.Circle((cx, cy), radius=1.8, facecolor=facecol,
                                        edgecolor='black', linewidth=0.5, zorder=20))

    # Legend
    legend_elements = [
        Line2D([0], [0], color='gray', linestyle='-', linewidth=0.8, label='Road edge'),
        Line2D([0], [0], color='gray', linestyle='--', linewidth=0.6, label='Lane divider'),
        patches.Patch(facecolor='cornflowerblue', edgecolor='black', label='Vehicle'),
        patches.Circle((0,0), radius=1.8, facecolor='red', edgecolor='black', label='Red light'),
        patches.Circle((0,0), radius=1.8, facecolor='green', edgecolor='black', label='Green light'),
        patches.Circle((0,0), radius=1.8, facecolor='yellow', edgecolor='black', label='Yellow light')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.savefig(f"{output_prefix}_000.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{output_prefix}_000.pdf", bbox_inches='tight')
    print(f"✅ Static frame saved as {output_prefix}_000.png and {output_prefix}_000.pdf")
    plt.close(fig)

# ----------------------------------------------------------------------
# 9. Animation with blitting (fast!)
# ----------------------------------------------------------------------
def animate_frames(frames_data, road_network, interval=50, save_mp4=False):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-320, 320)
    ax.set_ylim(-320, 320)
    ax.set_aspect('equal')
    ax.set_title("Traffic Intersection Simulation", fontsize=14, weight='bold')

    # Draw static background (once)
    draw_road_network(ax, road_network)

    # Prepare artists that will be updated
    vehicle_patches = []    # list of (shadow, body, headlights*2, taillights*2, label)
    tl_circles = {}         # dict mapping (road_id, lane_idx) -> circle patch

    # Pre-create all possible traffic light circles (but will be updated per frame)
    positions = road_network['tl_positions']
    tl_artists = []
    for road_id, pos_list in positions.items():
        for lane_idx, cx, cy in pos_list:
            circle = patches.Circle((cx, cy), radius=1.5, facecolor='gray',
                                    edgecolor='black', linewidth=0.5, zorder=20, visible=False)
            ax.add_patch(circle)
            tl_artists.append(circle)
            tl_circles[(road_id, lane_idx)] = circle

    # Pre-create vehicle artists (will be repositioned)
    # We'll create them dynamically each frame because number of vehicles may change.
    # But to use blitting, we need to keep a fixed list and toggle visibility.
    # Simpler: use blit=False but with static background. We'll stick with non-blit for simplicity,
    # but we've already saved the heavy road drawing. We'll just clear vehicles and traffic lights each frame.
    # Actually, blitting is complex with varying number of vehicles. We'll keep non-blit but with pre-drawn bg.

    # To avoid re-drawing roads, we can use blit=True and update only the dynamic artists.
    # We'll implement a hybrid: draw bg once, then use blitting with a fixed list of artists.
    # Since number of vehicles changes, we need to recreate them each frame and return updated artists.
    # That's possible with blit=True if we return the list of artists that have changed.
    # We'll do it properly:

    # Background artists are static; we'll store them and use blit=True.
    bg_artists = []
    for pts in road_network['road_fills']:
        bg_artists.append(ax.add_patch(patches.Polygon(pts, closed=True, facecolor='lightgray', edgecolor='none', zorder=1)))
    for x1, y1, x2, y2, style, lw, color in road_network['lane_boundaries']:
        bg_artists.append(ax.plot([x1, x2], [y1, y2], linestyle=style, linewidth=lw, color=color, zorder=2)[0])
    for pts in road_network['intersection_polys']:
        bg_artists.append(ax.add_patch(patches.Polygon(pts, closed=True, facecolor='lightgray', edgecolor='black', linewidth=1.0, zorder=5)))
    for pts in road_network['boundary_polys']:
        bg_artists.append(ax.add_patch(patches.Polygon(pts, closed=True, facecolor='white', edgecolor='black', linewidth=0.8, zorder=5)))

    # Frame counter text
    frame_text = ax.text(-310, -310, "", fontsize=8, zorder=30)

    def init():
        # Called once at start of animation
        for artist in bg_artists:
            artist.set_visible(True)
        for circle in tl_artists:
            circle.set_visible(False)
        frame_text.set_text("")
        return bg_artists + tl_artists + [frame_text]

    def update(frame_idx):
        frame = frames_data[frame_idx]

        # Update traffic lights
        tl_data = frame['traffic_lights']
        for circle in tl_artists:
            circle.set_visible(False)
        for road_id, colors in tl_data.items():
            if road_id not in positions:
                continue
            for lane_idx, cx, cy in positions[road_id]:
                key = (road_id, lane_idx)
                if key in tl_circles:
                    col = colors[lane_idx]
                    facecol = {'r': 'red', 'g': 'green', 'y': 'yellow'}.get(col, 'gray')
                    circle = tl_circles[key]
                    circle.set_facecolor(facecol)
                    circle.set_visible(True)

        # Remove old vehicle artists (we'll recreate them)
        for artist in reversed(ax.patches):
            if artist.get_zorder() in [9,10,11,12]:   # our vehicle layers
                artist.remove()
        for artist in reversed(ax.texts):
            if artist != frame_text:
                artist.remove()

        # Draw new vehicles
        for v in frame['vehicles']:
            draw_detailed_car(ax, v, show_label=False)

        # Update frame counter
        frame_text.set_text(f"Frame {frame_idx}")

        # Return all artists that were modified
        return bg_artists + tl_artists + [frame_text] + ax.patches + ax.texts

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames_data),
        init_func=init, blit=True, interval=interval, repeat=True, cache_frame_data=False
    )

    if save_mp4:
        print("💾 Saving animation as MP4...")
        ani.save('traffic_simulation.mp4', writer='ffmpeg', dpi=100)
        print("✅ Saved traffic_simulation.mp4")

    plt.show()

# ----------------------------------------------------------------------
# 10. Main with command line arguments
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Visualise traffic intersection from JSON roadnet and replay.")
    parser.add_argument('--roadnet', type=str, default='C:\\Users\\asdal\\replay\\roadnet.json',
                        help='Path to roadnet.json')
    parser.add_argument('--replay', type=str, default='C:\\Users\\asdal\\replay\\replay.txt',
                        help='Path to replay.txt')
    parser.add_argument('--output', type=str, default='frame',
                        help='Prefix for static output files')
    parser.add_argument('--save', action='store_true',
                        help='Save animation as MP4')
    parser.add_argument('--interval', type=int, default=50,
                        help='Frame interval in ms')
    args = parser.parse_args()

    # Check files exist
    for f in [args.roadnet, args.replay]:
        if not os.path.exists(f):
            print(f"❌ File not found: {f}")
            return

    print("📁 Parsing roadnet.json ...")
    nodes, edges = parse_roadnet(args.roadnet)
    print(f"✅ Loaded {len(nodes)} intersections, {len(edges)} road edges.")

    print("🎨 Pre‑computing road geometry...")
    road_network = prepare_road_network(nodes, edges)

    print_road_summary(nodes, edges, road_network['tl_positions'])

    print("📁 Parsing replay.txt ...")
    frames_data = parse_replay(args.replay)
    print(f"✅ Loaded {len(frames_data)} frames.")

    if not frames_data:
        print("❌ No frames found in replay file. Exiting.")
        return

    for i, f in enumerate(frames_data):
        f['index'] = i

    print("🖼️  Generating static publication frame...")
    plot_static_frame(frames_data[0], road_network, args.output)

    print("🎬 Starting animation...")
    animate_frames(frames_data, road_network, interval=args.interval, save_mp4=args.save)

if __name__ == '__main__':
    main()