import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from collections import defaultdict
import math

# Optional dependencies – graceful fallback
try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = False   # we don't use tqdm, but keep the check
except ImportError:
    pass

# ----------------------------------------------------------------------
# 1. Parse road network from your roadnet_3_4.json
# ----------------------------------------------------------------------
def parse_roadnet_3_4(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    intersections = data['intersections']
    roads = data['roads']

    # ---- nodes (intersections) ----
    nodes = {}
    for inter in intersections:
        nid = inter['id']
        pt = tuple(inter['point'].values())  # (x, y)
        width = inter['width']
        virtual = inter.get('virtual', False)

        # Create an outline for the intersection
        if virtual:
            # boundary node – small rectangle
            w = 2.0
            x, y = pt
            outline = [(x - w/2, y - w/2), (x + w/2, y - w/2),
                       (x + w/2, y + w/2), (x - w/2, y + w/2)]
        else:
            # central intersection – use width to create an octagon
            w = width / 2.0
            d = w * 0.414
            x, y = pt
            outline = [
                (x - w, y - d), (x - w, y + d),
                (x - d, y + w), (x + d, y + w),
                (x + w, y + d), (x + w, y - d),
                (x + d, y - w), (x - d, y - w)
            ]

        # ✅ FIXED: Preserve ALL original intersection data
        node_data = {
            'point': pt,
            'virtual': virtual,
            'outline': outline,
            'width': width
        }
        
        # Copy trafficLight data if it exists
        if 'trafficLight' in inter:
            node_data['trafficLight'] = inter['trafficLight']
        
        # Copy roadLinks if they exist (needed for traffic light mapping)
        if 'roadLinks' in inter:
            node_data['roadLinks'] = inter['roadLinks']
        
        nodes[nid] = node_data

    # ---- edges (roads) ----
    edges = {}
    for road in roads:
        eid = road['id']
        points = [tuple(p.values()) for p in road['points']]
        lanes_info = road['lanes']
        nLane = len(lanes_info)
        laneWidths = [lane['width'] for lane in lanes_info]
        total_width = sum(laneWidths)
        start_inter = road['startIntersection']
        end_inter = road['endIntersection']

        edges[eid] = {
            'from': start_inter,
            'to': end_inter,
            'points': points,
            'nLane': nLane,
            'laneWidths': laneWidths,
            'total_width': total_width
        }

    return nodes, edges


# ----------------------------------------------------------------------
# 2. Parse flow file
# ----------------------------------------------------------------------
def parse_flow(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    flows = []
    for entry in data:
        flows.append({
            'vehicle': entry['vehicle'],
            'route': entry['route'],
            'interval': entry['interval'],
            'startTime': entry['startTime'],
            'endTime': entry['endTime']
        })
    return flows

# ----------------------------------------------------------------------
# 3. Geometry helpers (same as original, slightly adapted)
# ----------------------------------------------------------------------
def get_boundary_attach_point(node):
    """For boundary (virtual) nodes, return the point on its outline that faces inward."""
    pt = node['point']
    x, y = pt
    outline = node['outline']
    if x == -300:   # left boundary – attach at right edge
        return (max(p[0] for p in outline), y)
    elif x == 300:  # right boundary – attach at left edge
        return (min(p[0] for p in outline), y)
    elif y == -300: # bottom boundary – attach at top edge
        return (x, max(p[1] for p in outline))
    elif y == 300:  # top boundary – attach at bottom edge
        return (x, min(p[1] for p in outline))
    else:
        return pt

def get_intersection_stop_points(intersection_outline, center_point):
    """Return four points on the outline where cardinal roads meet."""
    xs = [p[0] for p in intersection_outline]
    ys = [p[1] for p in intersection_outline]
    return {
        'east':  (max(xs), center_point[1]),
        'west':  (min(xs), center_point[1]),
        'north': (center_point[0], max(ys)),
        'south': (center_point[0], min(ys))
    }

def get_direction_vector(from_point, to_point):
    dx = to_point[0] - from_point[0]
    dy = to_point[1] - from_point[1]
    if abs(dx) > abs(dy):
        return 'east' if dx > 0 else 'west'
    else:
        return 'north' if dy > 0 else 'south'





# ----------------------------------------------------------------------
# 4. Pre‑compute all static road geometry
# ----------------------------------------------------------------------
def prepare_road_network(nodes, edges):
    road_fills_data = []
    lane_boundaries = []
    intersection_polys_data = []
    boundary_polys_data = []
    tl_positions = {}          # key: exit edge ID, value: list of (lane_idx, x, y)

    # The only non‑virtual intersection with traffic lights in your network is intersection_1_1
    # We need to identify the central intersection. In roadnet_3_4.json it is intersection_1_1.
    # We'll search for the intersection with width > 0 and that has trafficLight defined.
    # Find the central intersection with traffic lights
    center_node_id = None
    for nid, node in nodes.items():
        if not node['virtual'] and 'trafficLight' in node:
            center_node_id = nid
            break
    
    if center_node_id is None:
        raise ValueError("No intersection with traffic lights found!")
    
    center_node = nodes[center_node_id]
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
        if from_id == center_node_id:
            dir_from = get_direction_vector(center_point, to_node['point'])
            start = stop_points[dir_from]
        else:
            start = get_boundary_attach_point(from_node)

        if to_id == center_node_id:
            dir_to = get_direction_vector(center_point, from_node['point'])
            end = stop_points[dir_to]
        else:
            end = get_boundary_attach_point(to_node)

        # 1. Road surface polygon
        sl = (start[0] + half_width * perp_x, start[1] + half_width * perp_y)
        sr = (start[0] - half_width * perp_x, start[1] - half_width * perp_y)
        er = (end[0]   - half_width * perp_x, end[1]   - half_width * perp_y)
        el = (end[0]   + half_width * perp_x, end[1]   + half_width * perp_y)
        road_fills_data.append([sl, sr, er, el])

        # 2. Lane boundaries
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

        # 3. Traffic light positions – only for roads that *enter* the central intersection
        if to_id == center_node_id:
            stop_point = end
            cum = 0.0
            positions = []
            for i, w in enumerate(lane_widths):
                centre_off = half_width - cum - w/2.0
                cx = stop_point[0] + centre_off * perp_x
                cy = stop_point[1] + centre_off * perp_y
                positions.append((i, cx, cy))
                cum += w

            # Find the corresponding exit road (from center to same boundary node)
            boundary_node_id = from_id   # because to_id == center_node_id, the road enters from boundary
            exit_id = None
            for cand_id, cand in edges.items():
                if cand['from'] == center_node_id and cand['to'] == boundary_node_id:
                    exit_id = cand_id
                    break
            if exit_id:
                tl_positions[exit_id] = positions

    # 4. Intersection polygons (central only) and boundary polygons
    for nid, node in nodes.items():
        outline = node['outline']
        if nid == center_node_id:
            intersection_polys_data.append(outline)
        else:
            boundary_polys_data.append(outline)

    return {
        'road_fills': road_fills_data,
        'lane_boundaries': lane_boundaries,
        'intersection_polys': intersection_polys_data,
        'boundary_polys': boundary_polys_data,
        'tl_positions': tl_positions,
        'stop_points': stop_points
    }

    

# ----------------------------------------------------------------------
# 5. Console summary (same as original, adapted)
# ----------------------------------------------------------------------
def print_road_summary(nodes, edges, tl_positions):
    print("\n" + "="*60)
    print("              ROAD NETWORK STATICS".center(60))
    print("="*60)

    print("\n[ Intersections ]")
    inter_data = []
    for nid, node in nodes.items():
        inter_data.append([nid, "yes" if node['virtual'] else "no", node['point'], len(node['outline'])])
    if TABULATE_AVAILABLE:
        print(tabulate(inter_data, headers=["ID", "Virtual", "Center", "Vertices"], tablefmt="grid"))
    else:
        for row in inter_data:
            print(f"  {row[0]:<20} {row[1]:<6} {row[2]}  ({row[3]} pts)")

    print("\n[ Road Edges ]")
    edge_data = []
    for eid, e in edges.items():
        edge_data.append([eid, e['from'], e['to'], e['nLane'], e['laneWidths'], f"{e['total_width']:.1f} m"])
    if TABULATE_AVAILABLE:
        print(tabulate(edge_data, headers=["Edge ID", "From", "To", "#Lanes", "Lane widths", "Total width"], tablefmt="grid"))
    else:
        for row in edge_data:
            print(f"  {row[0]:<15} {row[1]:<18} {row[2]:<18} {row[3]}  {row[4]}  {row[5]}")

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
# 6. Vehicle drawing – detailed, with stable colour and dynamic scaling (same as original)
# ----------------------------------------------------------------------
def get_vehicle_color(name):
    """Deterministic stable colour per vehicle ID."""
    if 'vehicle' in name:
        # Extract the numeric part after the last underscore
        parts = name.split('_')
        if len(parts) > 1 and parts[-1].isdigit():
            hash_val = int(parts[-1])
        else:
            hash_val = abs(hash(name)) % 1000
    else:
        hash_val = abs(hash(name)) % 1000
    return plt.cm.tab20(hash_val % 20)


def draw_detailed_car(ax, v, show_label=False):
    x, y, angle = v['x'], v['y'], v['angle']
    length, width = v['length'], v['width']
    name = v['name']
    color = get_vehicle_color(name)

    # Shadow
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

    # Lights
    half_l = length / 2
    half_w = width / 2
    margin = min(length, width) * 0.1
    corners_rel = [
        (half_l - margin, -half_w + margin),  # FR
        (half_l - margin, half_w - margin),   # FL
        (-half_l + margin, -half_w + margin), # RR
        (-half_l + margin, half_w - margin)   # RL
    ]
    s, c = np.sin(angle), np.cos(angle)
    corners_abs = []
    for px, py in corners_rel:
        xnew = x + px * c - py * s
        ynew = y + px * s + py * c
        corners_abs.append((xnew, ynew))

    light_radius = min(length, width) * 0.12
    ax.add_patch(patches.Circle(corners_abs[0], light_radius, facecolor='yellow', edgecolor='none', zorder=11))
    ax.add_patch(patches.Circle(corners_abs[1], light_radius, facecolor='yellow', edgecolor='none', zorder=11))
    ax.add_patch(patches.Circle(corners_abs[2], light_radius, facecolor='red', edgecolor='none', zorder=11))
    ax.add_patch(patches.Circle(corners_abs[3], light_radius, facecolor='red', edgecolor='none', zorder=11))

    if show_label:
        ax.text(x, y + width/2 + 1.5, name, fontsize=6, ha='center', zorder=12)

# ----------------------------------------------------------------------
# 7. Simulation of traffic based on flow
# ----------------------------------------------------------------------
class TrafficSimulator:
    
    def __init__(self, nodes, edges, road_network, flows, dt=0.1):
        self.nodes = nodes
        self.edges = edges
        self.road_network = road_network
        self.flows = flows
        self.dt = dt
        self.time = 0.0
        self.next_vehicle_id = 0
        self.active_vehicles = []

        # Per‑flow spawn scheduling
        self.flow_spawns = []   # list of [next_spawn_time, flow_idx]
        for flow_idx, flow in enumerate(flows):
            next_time = flow['startTime']
            self.flow_spawns.append([next_time, flow_idx])

        # ✅ FIXED: Only use intersection_1_1 for traffic lights
        if 'intersection_1_1' not in self.nodes:
            raise ValueError("intersection_1_1 not found in nodes!")
        
        inter = self.nodes['intersection_1_1']
        if 'trafficLight' not in inter:
            raise ValueError("intersection_1_1 has no trafficLight data!")
        
        self.intersection_light = inter['trafficLight']
        self.light_cycle_time = 0.0
        self.current_phase = 0
        self.phase_durations = [phase['time'] for phase in self.intersection_light['lightphases']]

        # Pre‑compute road lengths for each lane
        self.lane_paths = self._precompute_lane_paths()

    def _precompute_lane_paths(self):
        """Store (start_x, start_y, end_x, end_y, length) for every (road_id, lane_idx)."""
        lane_paths = {}
        center_node_id = [nid for nid, n in self.nodes.items() if not n['virtual']][0]
        stop_points = self.road_network['stop_points']

        for eid, e in self.edges.items():
            total_width = e['total_width']
            half_width = total_width / 2.0
            lane_widths = e['laneWidths']

            # Raw direction
            pt_start = e['points'][0]
            pt_end = e['points'][1]
            dx = pt_end[0] - pt_start[0]
            dy = pt_end[1] - pt_start[1]
            length_raw = np.hypot(dx, dy)
            if length_raw == 0:
                continue
            dx_n, dy_n = dx / length_raw, dy / length_raw
            perp_x, perp_y = -dy_n, dx_n

            # Adjusted start / end points (same as in prepare_road_network)
            from_node = self.nodes[e['from']]
            to_node = self.nodes[e['to']]

            if e['from'] == center_node_id:
                dir_from = get_direction_vector(self.nodes[center_node_id]['point'], to_node['point'])
                start = stop_points[dir_from]
            else:
                start = get_boundary_attach_point(from_node)

            if e['to'] == center_node_id:
                dir_to = get_direction_vector(self.nodes[center_node_id]['point'], from_node['point'])
                end = stop_points[dir_to]
            else:
                end = get_boundary_attach_point(to_node)

            # Road length = distance between adjusted points
            road_length = np.hypot(end[0] - start[0], end[1] - start[1])

            # For each lane, compute lane centre line
            cum = 0.0
            for lane_idx, w in enumerate(lane_widths):
                centre_off = half_width - cum - w / 2.0
                start_lane = (start[0] + centre_off * perp_x, start[1] + centre_off * perp_y)
                end_lane   = (end[0]   + centre_off * perp_x, end[1]   + centre_off * perp_y)
                lane_paths[(eid, lane_idx)] = (
                    start_lane[0], start_lane[1],
                    end_lane[0],   end_lane[1],
                    road_length
                )
                cum += w
        return lane_paths

    def step(self):
        self.time += self.dt
        self.light_cycle_time += self.dt

        # Update traffic light phase
        total_cycle = sum(self.phase_durations)
        if self.light_cycle_time >= total_cycle:
            self.light_cycle_time = 0.0
        cum = 0.0
        for i, dur in enumerate(self.phase_durations):
            if self.light_cycle_time < cum + dur:
                self.current_phase = i
                break
            cum += dur

        # Spawn new vehicles (dynamic)
        for spawn in self.flow_spawns:
            next_time, flow_idx = spawn
            if self.time >= next_time:
                flow = self.flows[flow_idx]
                self._spawn_vehicle(flow)
                # Schedule next spawn
                next_time += flow['interval']
                if flow['endTime'] < 0 or next_time <= flow['endTime']:
                    spawn[0] = next_time
                else:
                    # Stop spawning this flow
                    spawn[0] = float('inf')

        # Update vehicle positions
        for v in self.active_vehicles[:]:
            self._update_vehicle(v)
            # Remove if reached end of route
            if v['route_index'] >= len(v['route']) - 1 and v['dist'] >= v['road_length']:
                self.active_vehicles.remove(v)

    def _spawn_vehicle(self, flow):
        route = flow['route']
        if not route:
            return
        road_id = route[0]
        # Choose lane (middle)
        road = self.edges[road_id]
        lane_idx = road['nLane'] // 2

        # Get pre‑computed lane path
        key = (road_id, lane_idx)
        if key not in self.lane_paths:
            print(f"Warning: no lane path for {key}")
            return
        sx, sy, ex, ey, road_length = self.lane_paths[key]

        # Initial angle = direction of road
        angle = math.atan2(ey - sy, ex - sx)

        veh = {
            'name': f'vehicle_{self.next_vehicle_id}',
            'x': sx,
            'y': sy,
            'angle': angle,
            'length': flow['vehicle']['length'],
            'width': flow['vehicle']['width'],
            'max_speed': flow['vehicle']['maxSpeed'],
            'speed': flow['vehicle']['maxSpeed'],
            'route': route,
            'route_index': 0,
            'dist': 0.0,
            'road_length': road_length,
            'lane_idx': lane_idx,
            'status': 1
        }
        self.active_vehicles.append(veh)
        self.next_vehicle_id += 1

    def _update_vehicle(self, v):
        road_id = v['route'][v['route_index']]
        road = self.edges[road_id]
        key = (road_id, v['lane_idx'])
        if key not in self.lane_paths:
            # Fallback – remove vehicle
            self.active_vehicles.remove(v)
            return
        sx, sy, ex, ey, road_length = self.lane_paths[key]

        # Traffic light logic (simplified)
        stop = False
        if v['route_index'] + 1 < len(v['route']) and v['route'][v['route_index']+1] == 'intersection_1_1':
            # Vehicle is about to enter intersection
            boundary_node_id = road['from'] if road['to'] == 'intersection_1_1' else road['to']
            exit_id = None
            for cand_id, cand in self.edges.items():
                if cand['from'] == 'intersection_1_1' and cand['to'] == boundary_node_id:
                    exit_id = cand_id
                    break
            if exit_id and exit_id in self.road_network['tl_positions']:
                central_node = self.nodes['intersection_1_1']
                # Find roadLink index for this exit road
                road_link_idx = None
                for rl_idx, rl in enumerate(central_node['roadLinks']):
                    if rl['startRoad'] == 'intersection_1_1' and rl['endRoad'] == boundary_node_id:
                        road_link_idx = rl_idx
                        break
                if road_link_idx is not None:
                    available = central_node['trafficLight']['lightphases'][self.current_phase]['availableRoadLinks']
                    if road_link_idx not in available:
                        # Red light – stop near stop line
                        remaining = v['road_length'] - v['dist']
                        if remaining < 5.0:
                            v['speed'] = 0.0
                            stop = True

        if not stop:
            # Accelerate
            if v['speed'] < v['max_speed']:
                v['speed'] = min(v['max_speed'], v['speed'] + 2.0 * self.dt)

        # Move
        ds = v['speed'] * self.dt
        v['dist'] += ds
        frac = min(v['dist'] / v['road_length'], 1.0)
        v['x'] = sx + (ex - sx) * frac
        v['y'] = sy + (ey - sy) * frac

        # Switch to next road if end reached
        if v['dist'] >= v['road_length']:
            v['route_index'] += 1
            if v['route_index'] < len(v['route']):
                next_road_id = v['route'][v['route_index']]
                # Keep same lane index if possible, else adjust
                next_road = self.edges[next_road_id]
                next_lane_idx = min(v['lane_idx'], next_road['nLane'] - 1)
                v['lane_idx'] = next_lane_idx
                next_key = (next_road_id, next_lane_idx)
                if next_key in self.lane_paths:
                    _, _, _, _, next_len = self.lane_paths[next_key]
                    v['road_length'] = next_len
                    v['dist'] = 0.0
                else:
                    # No path – remove vehicle
                    self.active_vehicles.remove(v)

    def get_frame_data(self):
        """Return format compatible with original replay frame."""
        vehicles = []
        for v in self.active_vehicles:
            vehicles.append({
                'x': v['x'],
                'y': v['y'],
                'angle': v['angle'],
                'name': v['name'],
                'status': 1,
                'length': v['length'],
                'width': v['width']
            })

        # Traffic lights – only for intersection_1_1
        tl_data = {}
        central_node = self.nodes['intersection_1_1']
        light_phase = central_node['trafficLight']['lightphases'][self.current_phase]
        available_indices = light_phase['availableRoadLinks']
        
        # Build mapping from roadLink index to exit road ID
        idx_to_exit = {}
        for rl_idx, rl in enumerate(central_node['roadLinks']):
            if rl['startRoad'] == 'intersection_1_1':
                idx_to_exit[rl_idx] = rl['endRoad']
        
        # For each exit road that has traffic lights (positions), determine color
        for exit_id, positions in self.road_network['tl_positions'].items():
            # Find the roadLink index that corresponds to this exit
            link_idx = None
            for idx, end in idx_to_exit.items():
                if end == exit_id:
                    link_idx = idx
                    break
            if link_idx is None:
                continue
            # Determine color for each lane
            colors = []
            for lane_idx, _, _ in positions:
                if link_idx in available_indices:
                    colors.append('g')  # green
                else:
                    colors.append('r')  # red
            tl_data[exit_id] = colors
        
        return {'vehicles': vehicles, 'traffic_lights': tl_data}


# ----------------------------------------------------------------------
# 8. Common road drawing function (same as original)
# ----------------------------------------------------------------------
def draw_road_network(ax, rn):
    for pts in rn['road_fills']:
        ax.add_patch(patches.Polygon(pts, closed=True, facecolor='lightgray', edgecolor='none', zorder=1))
    for x1, y1, x2, y2, style, lw, color in rn['lane_boundaries']:
        ax.plot([x1, x2], [y1, y2], linestyle=style, linewidth=lw, color=color, zorder=2)
    for pts in rn['intersection_polys']:
        ax.add_patch(patches.Polygon(pts, closed=True, facecolor='lightgray', edgecolor='black', linewidth=1.0, zorder=5))
    for pts in rn['boundary_polys']:
        ax.add_patch(patches.Polygon(pts, closed=True, facecolor='white', edgecolor='black', linewidth=0.8, zorder=5))

# ----------------------------------------------------------------------
# 9. Static publication frame (first frame of simulation)
# ----------------------------------------------------------------------
def plot_static_frame(frame, road_network, output_prefix="frame"):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(-320, 320)
    ax.set_ylim(-320, 320)
    ax.set_aspect('equal')
    ax.set_title("Intersection snapshot – Frame 0", fontsize=16, weight='bold')

    draw_road_network(ax, road_network)

    for v in frame['vehicles']:
        draw_detailed_car(ax, v, show_label=True)

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
# 10. Animation with blitting (same as original, but using simulator)
# ----------------------------------------------------------------------
def animate_simulation(simulator, road_network, interval=50, save_mp4=False, total_frames=600):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-320, 320)
    ax.set_ylim(-320, 320)
    ax.set_aspect('equal')
    ax.set_title("Traffic Intersection Simulation (Flow)", fontsize=14, weight='bold')

    # Draw static background (once)
    draw_road_network(ax, road_network)

    # Pre-create traffic light circles
    tl_positions = road_network['tl_positions']
    tl_circles = {}
    tl_artists = []
    for road_id, pos_list in tl_positions.items():
        for lane_idx, cx, cy in pos_list:
            circle = patches.Circle((cx, cy), radius=1.5, facecolor='gray',
                                    edgecolor='black', linewidth=0.5, zorder=20, visible=False)
            ax.add_patch(circle)
            tl_artists.append(circle)
            tl_circles[(road_id, lane_idx)] = circle

    # Frame counter text
    frame_text = ax.text(-310, -310, "", fontsize=8, zorder=30)

    # Background artists for blitting
    bg_artists = []
    for pts in road_network['road_fills']:
        bg_artists.append(ax.add_patch(patches.Polygon(pts, closed=True, facecolor='lightgray', edgecolor='none', zorder=1)))
    for x1, y1, x2, y2, style, lw, color in road_network['lane_boundaries']:
        bg_artists.append(ax.plot([x1, x2], [y1, y2], linestyle=style, linewidth=lw, color=color, zorder=2)[0])
    for pts in road_network['intersection_polys']:
        bg_artists.append(ax.add_patch(patches.Polygon(pts, closed=True, facecolor='lightgray', edgecolor='black', linewidth=1.0, zorder=5)))
    for pts in road_network['boundary_polys']:
        bg_artists.append(ax.add_patch(patches.Polygon(pts, closed=True, facecolor='white', edgecolor='black', linewidth=0.8, zorder=5)))

    def init():
        for artist in bg_artists:
            artist.set_visible(True)
        for circle in tl_artists:
            circle.set_visible(False)
        frame_text.set_text("")
        return bg_artists + tl_artists + [frame_text]

    def update(frame_idx):
        # Advance simulation for one time step
        simulator.step()
        frame = simulator.get_frame_data()

        # Update traffic lights
        tl_data = frame['traffic_lights']
        for circle in tl_artists:
            circle.set_visible(False)
        for road_id, colors in tl_data.items():
            if road_id not in tl_positions:
                continue
            for lane_idx, cx, cy in tl_positions[road_id]:
                key = (road_id, lane_idx)
                if key in tl_circles:
                    col = colors[lane_idx]
                    facecol = {'r': 'red', 'g': 'green', 'y': 'yellow'}.get(col, 'gray')
                    circle = tl_circles[key]
                    circle.set_facecolor(facecol)
                    circle.set_visible(True)

        # Remove old vehicle artists
        for artist in reversed(ax.patches):
            if artist.get_zorder() in [9,10,11,12]:
                artist.remove()
        for artist in reversed(ax.texts):
            if artist != frame_text:
                artist.remove()

        # Draw new vehicles
        for v in frame['vehicles']:
            draw_detailed_car(ax, v, show_label=False)

        # Update frame counter
        frame_text.set_text(f"Time: {simulator.time:.1f}s")

        # Return all artists that were modified
        return bg_artists + tl_artists + [frame_text] + ax.patches + ax.texts

    ani = animation.FuncAnimation(
        fig, update, frames=range(total_frames),
        init_func=init, blit=True, interval=interval, repeat=True, cache_frame_data=False
    )

    if save_mp4:
        print("💾 Saving animation as MP4...")
        ani.save('traffic_simulation_flow.mp4', writer='ffmpeg', dpi=100)
        print("✅ Saved traffic_simulation_flow.mp4")

    plt.show()

# ----------------------------------------------------------------------
# 11. Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Visualise traffic from roadnet and flow JSON (3_4 format).")
    parser.add_argument('--roadnet', type=str, default='roadnet_3_4.json',
                        help='Path to roadnet_3_4.json')
    parser.add_argument('--flow', type=str, default='flow_3_4.json',
                        help='Path to flow_3_4.json')
    parser.add_argument('--output', type=str, default='frame',
                        help='Prefix for static output files')
    parser.add_argument('--save', action='store_true',
                        help='Save animation as MP4')
    parser.add_argument('--interval', type=int, default=50,
                        help='Frame interval in ms (animation speed)')
    parser.add_argument('--duration', type=float, default=60.0,
                        help='Simulation duration in seconds')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='Simulation time step (s)')
    args = parser.parse_args()

    # Check files exist
    for f in [args.roadnet, args.flow]:
        if not os.path.exists(f):
            print(f"❌ File not found: {f}")
            return

    print("📁 Parsing roadnet_3_4.json ...")
    nodes, edges = parse_roadnet_3_4(args.roadnet)
    print(f"✅ Loaded {len(nodes)} intersections, {len(edges)} road edges.")

    print("📁 Parsing flow_3_4.json ...")
    flows = parse_flow(args.flow)
    print(f"✅ Loaded {len(flows)} flow entries.")

    print("🎨 Pre‑computing road geometry...")
    road_network = prepare_road_network(nodes, edges)

    print_road_summary(nodes, edges, road_network['tl_positions'])

    print("🚦 Initialising simulator...")
    sim = TrafficSimulator(nodes, edges, road_network, flows, dt=args.dt)

    print("🖼️  Generating static publication frame (t=0)...")
    frame0 = sim.get_frame_data()
    plot_static_frame(frame0, road_network, args.output)

    total_frames = int(args.duration / args.dt)
    print(f"🎬 Starting animation ({total_frames} frames, {args.duration}s simulated)...")
    animate_simulation(sim, road_network, interval=args.interval,
                       save_mp4=args.save, total_frames=total_frames)

if __name__ == '__main__':
    main()