import json
import xml.etree.ElementTree as ET
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import shutil

# -------------------------
# Paths
# -------------------------
PROJECT_DIR = Path(r"C:\Users\asdal\replay_9_9")

ROADNET_FILE = PROJECT_DIR / "roadnet_9_9_turn.json"
FLOW_FILE    = PROJECT_DIR / "flow_9_9_turn.json"
NODES_FILE   = PROJECT_DIR / "nodes.nod.xml"
EDGES_FILE   = PROJECT_DIR / "edges.edg.xml"
NET_FILE     = PROJECT_DIR / "network.net.xml"
FCD_FILE     = PROJECT_DIR / "fcd.rou.xml"
CONFIG_FILE  = PROJECT_DIR / "replay.sumocfg"

SUMO_BIN = Path(r"C:\Program Files (x86)\Eclipse\Sumo\bin")

# -------------------------
# Helper
# -------------------------
def write_xml(file_path: Path, content: str):
    file_path.write_text(content)
    print(f"{file_path.name} written: {file_path}")

# -------------------------
# 1️⃣ Load CityFlow roadnet.json
# -------------------------
with ROADNET_FILE.open() as f:
    data = json.load(f)

# Detect format
if "static" in data:  # old format
    nodes_list = data["static"]["nodes"]
    edges_list = data["static"]["edges"]
elif "nodes" in data and "edges" in data:  # common new format
    nodes_list = data["nodes"]
    edges_list = data["edges"]
elif "intersections" in data and "roads" in data:  # CityFlow 2.x
    nodes_list = data["intersections"]
    edges_list = data["roads"]
else:
    raise KeyError("JSON format not recognized. Expected 'static', 'nodes/edges', or 'intersections/roads'.")

print(f"Loaded {len(nodes_list)} nodes and {len(edges_list)} edges.")

# -------------------------
# 2️⃣ Write nodes XML
# -------------------------
node_dict: Dict[str, Tuple[float, float]] = {}
nodes_xml = "<nodes>\n"
for node in nodes_list:
    nid = str(node["id"])
    x = float(node["point"]["x"])
    y = float(node["point"]["y"])
    node_dict[nid] = (x, y)
    nodes_xml += f'  <node id="{nid}" x="{x}" y="{y}" type="priority"/>\n'
nodes_xml += "</nodes>\n"
write_xml(NODES_FILE, nodes_xml)

# -------------------------
# 3️⃣ Write edges XML
# -------------------------
edges_xml = "<edges>\n"
for edge in edges_list:
    rid = str(edge["id"])
    frm = str(edge.get("from") or edge.get("startIntersection"))
    to = str(edge.get("to") or edge.get("endIntersection"))
    lanes = edge.get("nLane") or edge.get("numLanes", 1)
    edges_xml += f'  <edge id="{rid}" from="{frm}" to="{to}" numLanes="{lanes}" speed="13.9"/>\n'
edges_xml += "</edges>\n"
write_xml(EDGES_FILE, edges_xml)

# -------------------------
# 4️⃣ Run netconvert
# -------------------------
print("Creating SUMO network ...")
subprocess.run([
    SUMO_BIN / "netconvert.exe",
    "-n", str(NODES_FILE),
    "-e", str(EDGES_FILE),
    "-o", str(NET_FILE)
], check=True)
print("SUMO network created:", NET_FILE)

# -------------------------
# 5️⃣ Parse flow file (fixed for replay files)
# -------------------------
print("Parsing flow file ...")
FLOW_FILE_CONTENT = FLOW_FILE.read_text().strip()
vehicle_positions: Dict[str, List[Tuple[int, float, float, float, List[str]]]] = {}

if FLOW_FILE_CONTENT.startswith('<'):
    # XML format (SUMO routes)
    print("Detected XML format (SUMO routes)")
    shutil.copy(FLOW_FILE, FCD_FILE)
    tree = ET.parse(FLOW_FILE)
    root = tree.getroot()
    for vehicle in root.findall('.//vehicle'):
        vid = vehicle.get('id')
        depart = int(vehicle.get('depart', '0'))
        vehicle_positions[vid] = [(depart, 0.0, 0.0, 0.0, [])]
    print(f"Detected {len(vehicle_positions)} vehicles from XML flow file.")

else:
    # JSON format – try to parse
    print("Detected JSON format (CityFlow)")
    with FLOW_FILE.open() as f:
        try:
            flow_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in flow file: {e}")
            flow_data = None

    if flow_data is not None:
        vehicles_found = 0

        # Helper to extract vehicle info from a dict
        def extract_vehicle(vdict, default_time=0):
            vid = vdict.get("id") or vdict.get("vehicle") or vdict.get("ID")
            if not vid or not isinstance(vid, str):
                return None
            route_edges = vdict.get("route") or vdict.get("edges") or []
            start_time = vdict.get("startTime") or vdict.get("depart") or vdict.get("departure") or default_time
            try:
                start_time = int(float(start_time))
            except (TypeError, ValueError):
                start_time = default_time
            return vid, start_time, route_edges

        # Determine the structure of flow_data
        if isinstance(flow_data, list):
            if not flow_data:
                print("Flow file is an empty list.")
            else:
                first = flow_data[0]
                # Heuristic: if first item has keys typical of a frame (interval, time, vehicle list)...
                if isinstance(first, dict):
                    # Look for keys that indicate a frame
                    if any(k in first for k in ["interval", "time", "frame"]) or \
                       (("vehicle" in first or "vehicles" in first) and 
                        isinstance(first.get("vehicle") or first.get("vehicles"), (list, dict))):
                        print("Detected list of frames (CityFlow replay)")
                        for frame_idx, frame in enumerate(flow_data):
                            if not isinstance(frame, dict):
                                continue
                            # Get the vehicle list – could be under "vehicle" or "vehicles"
                            veh_list = frame.get("vehicle") or frame.get("vehicles")
                            if veh_list is None:
                                # Maybe the frame itself is a list of vehicles (rare)
                                if isinstance(frame, list):
                                    veh_list = frame
                                else:
                                    continue
                            # Ensure veh_list is a list
                            if isinstance(veh_list, dict):
                                veh_list = [veh_list]  # single vehicle
                            if not isinstance(veh_list, list):
                                continue
                            for v in veh_list:
                                if isinstance(v, dict):
                                    # Try to get vehicle id
                                    vid = v.get("id") or v.get("vehicle") or v.get("ID")
                                    if not vid or not isinstance(vid, str):
                                        continue
                                    x = float(v.get("x", 0))
                                    y = float(v.get("y", 0))
                                    angle = float(v.get("angle", 0))
                                    # Record first appearance only
                                    if vid not in vehicle_positions:
                                        vehicle_positions[vid] = [(frame_idx, x, y, angle, [])]
                                        vehicles_found += 1
                                elif isinstance(v, list) and len(v) >= 4:
                                    # [x, y, angle, id] format
                                    x, y, angle, vid = float(v[0]), float(v[1]), float(v[2]), str(v[3])
                                    if vid not in vehicle_positions:
                                        vehicle_positions[vid] = [(frame_idx, x, y, angle, [])]
                                        vehicles_found += 1
                    elif "id" in first or "route" in first:
                        print("Detected list of vehicle definitions")
                        for item in flow_data:
                            if isinstance(item, dict):
                                veh_info = extract_vehicle(item)
                                if veh_info:
                                    vid, start, route = veh_info
                                    vehicle_positions[vid] = [(start, 0.0, 0.0, 0.0, route)]
                                    vehicles_found += 1
                    else:
                        print("Unknown list structure – skipping")
                else:
                    print("First element is not a dict – cannot parse")
        elif isinstance(flow_data, dict):
            # ... (same as before) ...
            # Look for a key that contains a list of vehicles
            vehicle_container_keys = ["vehicles", "vehicle", "flow", "cars", "flows"]
            found_container = False
            for key in vehicle_container_keys:
                if key in flow_data and isinstance(flow_data[key], list):
                    print(f"Found vehicle list under key '{key}'")
                    for item in flow_data[key]:
                        if isinstance(item, dict):
                            veh_info = extract_vehicle(item)
                            if veh_info:
                                vid, start, route = veh_info
                                vehicle_positions[vid] = [(start, 0.0, 0.0, 0.0, route)]
                                vehicles_found += 1
                    found_container = True
                    break

            if not found_container:
                # Maybe the dict itself is keyed by vehicle ID
                print("No top-level vehicle list found; trying vehicle-ID dictionary")
                for vid, vdata in flow_data.items():
                    if isinstance(vdata, dict):
                        start = vdata.get("startTime") or vdata.get("depart") or 0
                        route = vdata.get("route") or vdata.get("edges") or []
                        try:
                            start = int(float(start))
                        except (TypeError, ValueError):
                            start = 0
                        vehicle_positions[vid] = [(start, 0.0, 0.0, 0.0, route)]
                        vehicles_found += 1

        else:
            print("Flow data is neither list nor dict – cannot parse.")

        print(f"Detected {vehicles_found} vehicles from JSON flow file.")

# -------------------------
# 6️⃣ Map network edges
# -------------------------
net_tree = ET.parse(NET_FILE)
net_root = net_tree.getroot()
edges_segments: List[Tuple[str, Tuple[float, float], Tuple[float, float]]] = []
for edge in net_root.findall('edge'):
    eid = edge.attrib['id']
    if 'from' not in edge.attrib or 'to' not in edge.attrib:
        continue
    frm_x, frm_y = node_dict[edge.attrib['from']]
    to_x, to_y = node_dict[edge.attrib['to']]
    edges_segments.append((eid, (frm_x, frm_y), (to_x, to_y)))

def closest_edge(x: float, y: float) -> str:
    min_dist = float("inf")
    best_edge = edges_segments[0][0]
    for eid, (x1, y1), (x2, y2) in edges_segments:
        px, py = x2 - x1, y2 - y1
        norm = px*px + py*py
        u = ((x - x1)*px + (y - y1)*py) / max(norm, 1e-6)
        u = max(0, min(1, u))
        closest_x, closest_y = x1 + u*px, y1 + u*py
        dist2 = (x - closest_x)**2 + (y - closest_y)**2
        if dist2 < min_dist:
            min_dist = dist2
            best_edge = eid
    return best_edge

# -------------------------
# 7️⃣ Create FCD vehicle file
# -------------------------
root = ET.Element("routes")
ET.SubElement(root, "vType", id="car", accel="2.6", decel="4.5", sigma="0.5", length="5", maxSpeed="13.9")

for vid, pos_list in vehicle_positions.items():
    start_time, x, y, angle, route_edges = pos_list[0]
    vehicle = ET.SubElement(root, "vehicle", id=vid, type="car", depart=str(start_time))
    route = ET.SubElement(vehicle, "route")
    if route_edges:
        route.set("edges", " ".join(route_edges))
    else:
        route.set("edges", closest_edge(x, y))

ET.ElementTree(root).write(FCD_FILE)
print("FCD vehicle file saved:", FCD_FILE)

# -------------------------
# 8️⃣ Create SUMO config
# -------------------------
num_frames = 0
if vehicle_positions:
    num_frames = max(pos[0] for pos_list in vehicle_positions.values() for pos in pos_list) + 1

with CONFIG_FILE.open("w") as f:
    f.write(f"""<configuration>
    <input>
        <net-file value="{NET_FILE}"/>
        <route-files value="{FCD_FILE}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{num_frames}"/>
    </time>
</configuration>""")
print("SUMO config saved:", CONFIG_FILE)

# -------------------------
# 9️⃣ Launch SUMO GUI
# -------------------------
subprocess.run([SUMO_BIN / "sumo-gui.exe", str(CONFIG_FILE)])