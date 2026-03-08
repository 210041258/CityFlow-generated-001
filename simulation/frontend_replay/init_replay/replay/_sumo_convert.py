import json
import xml.etree.ElementTree as ET
import subprocess
import os
from scipy.spatial import KDTree

# -------------------------
# User paths – edit these
# -------------------------
ROADNET_FILE = r"C:\Users\asdal\replay\roadnet.json"
NODES_FILE   = r"C:\Users\asdal\replay\nodes.nod.xml"
EDGES_FILE   = r"C:\Users\asdal\replay\edges.edg.xml"
NET_FILE     = r"C:\Users\asdal\replay\network.net.xml"
REPLAY_FILE  = r"C:\Users\asdal\replay\replay.txt"
FCD_FILE     = r"C:\Users\asdal\replay\fcd.rou.xml"
CONFIG_FILE  = r"C:\Users\asdal\replay\replay.sumocfg"

SUMO_BIN     = r"C:\Program Files (x86)\Eclipse\Sumo\bin"

# -------------------------
# 1️⃣ Convert CityFlow roadnet.json → nodes + edges
# -------------------------
with open(ROADNET_FILE) as f:
    data = json.load(f)

nodes_list = data["static"]["nodes"]
edges_list = data["static"]["edges"]

# Nodes
node_dict = {}  # node_id -> (x, y)
with open(NODES_FILE, "w") as f:
    f.write("<nodes>\n")
    for node in nodes_list:
        nid = node["id"]
        x, y = node["point"]
        node_dict[nid] = (x, y)
        f.write(f'<node id="{nid}" x="{x}" y="{y}" type="priority"/>\n')
    f.write("</nodes>\n")
print("Nodes written:", NODES_FILE)

# Edges
with open(EDGES_FILE, "w") as f:
    f.write("<edges>\n")
    for edge in edges_list:
        rid = edge["id"]
        frm = edge["from"]
        to = edge["to"]
        lanes = edge["nLane"]
        f.write(f'<edge id="{rid}" from="{frm}" to="{to}" numLanes="{lanes}" speed="13.9"/>\n')
    f.write("</edges>\n")
print("Edges written:", EDGES_FILE)

# -------------------------
# 2️⃣ Run netconvert → network.net.xml
# -------------------------
print("Creating SUMO network with netconvert ...")
subprocess.run([
    os.path.join(SUMO_BIN, "netconvert.exe"),
    "-n", NODES_FILE,
    "-e", EDGES_FILE,
    "-o", NET_FILE
], check=True)
print("SUMO network created:", NET_FILE)

# -------------------------
# 3️⃣ Parse replay.txt → vehicle positions
# -------------------------
print("Parsing replay.txt ...")
vehicle_positions = {}  # vid -> list of (frame, x, y, angle)
with open(REPLAY_FILE) as f:
    for frame_idx, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        parts = line.split(";")
        vehicles = parts[0]
        for v in vehicles.split(","):
            tokens = v.strip().split()
            if len(tokens) != 7:
                continue
            x = float(tokens[0])
            y = float(tokens[1])
            angle = float(tokens[2])
            vid = tokens[3]
            if vid not in vehicle_positions:
                vehicle_positions[vid] = []
            vehicle_positions[vid].append((frame_idx, x, y, angle))

print(f"Detected {len(vehicle_positions)} vehicles.")

# -------------------------
# 4️⃣ Load SUMO edges → KDTree for mapping (skip internal edges)
# -------------------------
print("Mapping vehicles to SUMO edges ...")
net_tree = ET.parse(NET_FILE)
net_root = net_tree.getroot()

edges_coords = []
edges_ids = []

for edge in net_root.findall('edge'):
    eid = edge.attrib['id']
    # Skip edges without 'from'/'to' (internal edges)
    if 'from' not in edge.attrib or 'to' not in edge.attrib:
        continue
    frm_node = edge.attrib['from']
    to_node = edge.attrib['to']
    frm_x, frm_y = node_dict[frm_node]
    to_x, to_y = node_dict[to_node]
    mx = (frm_x + to_x) / 2
    my = (frm_y + to_y) / 2
    edges_coords.append((mx, my))
    edges_ids.append(eid)

edge_tree = KDTree(edges_coords)

# -------------------------
# 5️⃣ Create FCD-style route file with mapped edges
# -------------------------
print("Creating FCD vehicle file ...")
root = ET.Element("routes")

# Vehicle type
vtype = ET.SubElement(root, "vType")
vtype.set("id", "car")
vtype.set("accel", "2.6")
vtype.set("decel", "4.5")
vtype.set("sigma", "0.5")
vtype.set("length", "5")
vtype.set("maxSpeed", "13.9")

for vid, pos_list in vehicle_positions.items():
    vehicle = ET.SubElement(root, "vehicle", id=vid, type="car", depart=str(pos_list[0][0]))
    route = ET.SubElement(vehicle, "route")
    x, y = pos_list[0][1], pos_list[0][2]
    _, idx = edge_tree.query((x, y))
    route.set("edges", edges_ids[idx])

tree = ET.ElementTree(root)
tree.write(FCD_FILE)
print("FCD vehicle file saved:", FCD_FILE)

# -------------------------
# 6️⃣ Create SUMO config
# -------------------------
print("Creating SUMO config ...")
with open(CONFIG_FILE, "w") as f:
    f.write(f"""<configuration>
<input>
    <net-file value="{NET_FILE}"/>
    <route-files value="{FCD_FILE}"/>
</input>
<time>
    <begin value="0"/>
    <end value="{len(next(iter(vehicle_positions.values())))}"/>
</time>
</configuration>""")
print("SUMO config saved:", CONFIG_FILE)

# -------------------------
# 7️⃣ Launch SUMO GUI
# -------------------------
print("Launching SUMO GUI ...")
subprocess.run([os.path.join(SUMO_BIN, "sumo-gui.exe"), CONFIG_FILE])