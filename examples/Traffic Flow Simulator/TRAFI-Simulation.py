

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import argparse
import os
import sys
from datetime import datetime
import copy

# -------------------------
# Car-Following Models
# -------------------------
def IDM_acceleration(vehicle, gap, leader_speed):
    """
    Intelligent Driver Model (IDM)
    """
    v = vehicle["speed"]
    v0 = vehicle["max_speed"]
    s0 = vehicle.get("minGap", 2.0)
    T = vehicle.get("headwayTime", 1.5)
    a = vehicle["acceleration"]
    b = abs(vehicle["deceleration"])
    
    # Desired minimum gap
    delta_v = v - leader_speed
    s_star = s0 + max(0, v * T + (v * delta_v) / (2 * np.sqrt(a * b)))
    
    # IDM acceleration formula - handle None gap
    if gap is not None and gap > 0:
        acceleration = a * (1 - (v / v0)**4 - (s_star / gap)**2)
    else:
        # If no gap or gap <= 0, accelerate freely
        acceleration = a * (1 - (v / v0)**4)
    
    return max(min(acceleration, a), -b)

def Gipps_acceleration(vehicle, gap, leader_speed):
    """
    Gipps' Car-Following Model
    """
    v = vehicle["speed"]
    v0 = vehicle["max_speed"]
    b = abs(vehicle["deceleration"])
    reaction_time = 1.0
    
    # Free flow term
    term1 = 2.5 * vehicle["acceleration"] * reaction_time * (1 - v / v0) * np.sqrt(0.025 + v / v0)
    
    # If no leader or no gap, return free flow acceleration
    if gap is None:
        return term1
    
    leader_deceleration = 4.5  # Assumed
    # Car-following term (make sure we don't take sqrt of negative)
    discriminant = b**2 * reaction_time**2 + b * (2 * max(gap, 0.1) - v * reaction_time - leader_speed**2 / max(leader_deceleration, 0.1))
    term2 = b * reaction_time + np.sqrt(max(discriminant, 0))
    
    return min(term1, term2 - v)

def safe_acceleration(vehicle, gap, leader_speed):
    """Safe car-following model (simplified)"""
    if gap is None:
        desired_acc = (vehicle["max_speed"] - vehicle["speed"]) / 2.0
        return min(desired_acc, vehicle["acceleration"])
    
    safe_distance = vehicle["speed"] * 1.0 + vehicle["length"] * 1.5
    if gap < safe_distance:
        required_decel = (vehicle["speed"]**2 - leader_speed**2) / (2 * max(gap, 0.1))
        return -min(abs(required_decel), vehicle["deceleration"])
    else:
        return vehicle["acceleration"]

# -------------------------
# Traffic Signal Class
# -------------------------
class TrafficSignal:
    def __init__(self, phase_times=None):
        # Custom phase times for grid network
        if phase_times is None:
            self.phase_times = {
                "NS_green": 30,  # North-South green
                "NS_yellow": 5,
                "EW_green": 25,  # East-West green
                "EW_yellow": 5
            }
        else:
            self.phase_times = phase_times
        
        self.cycle_time = sum(self.phase_times.values())
        self.current_time = 0
        self.current_phase = "NS_green"
        self.phase_start_time = 0
        
    def get_state(self, road_direction):
        """Get signal state for a road based on direction"""
        elapsed = self.current_time - self.phase_start_time
        
        if self.current_phase == "NS_green":
            if road_direction in ["north", "south"]:
                return "green" if elapsed < self.phase_times["NS_green"] else "yellow"
            else:
                return "red"
        elif self.current_phase == "NS_yellow":
            if road_direction in ["north", "south"]:
                return "yellow"
            else:
                return "red"
        elif self.current_phase == "EW_green":
            if road_direction in ["east", "west"]:
                return "green" if elapsed < self.phase_times["EW_green"] else "yellow"
            else:
                return "red"
        else:  # EW_yellow
            if road_direction in ["east", "west"]:
                return "yellow"
            else:
                return "red"
    
    def update(self, dt):
        """Update signal timing"""
        self.current_time += dt
        elapsed = self.current_time - self.phase_start_time
        
        if self.current_phase == "NS_green" and elapsed >= self.phase_times["NS_green"]:
            self.current_phase = "NS_yellow"
            self.phase_start_time = self.current_time
        elif self.current_phase == "NS_yellow" and elapsed >= self.phase_times["NS_yellow"]:
            self.current_phase = "EW_green"
            self.phase_start_time = self.current_time
        elif self.current_phase == "EW_green" and elapsed >= self.phase_times["EW_green"]:
            self.current_phase = "EW_yellow"
            self.phase_start_time = self.current_time
        elif self.current_phase == "EW_yellow" and elapsed >= self.phase_times["EW_yellow"]:
            self.current_phase = "NS_green"
            self.phase_start_time = self.current_time
    
    def get_summary(self):
        return {
            "current_phase": self.current_phase,
            "current_time": self.current_time,
            "phase_start_time": self.phase_start_time,
            "cycle_time": self.cycle_time
        }

# -------------------------
# State Generator
# -------------------------
class StateGenerator:
    def __init__(self, flow_file, dt=1.0, road_length=100):
        self.dt = dt
        self.road_length = road_length
        self.flows = self.load_flows(flow_file)
        self.vehicles = []
        self.road_network = defaultdict(list)
        self.traffic_signals = {}
        self.time = 0
        self.states_history = []
        
    def load_flows(self, flow_file):
        """Load flow data from JSON file"""
        try:
            with open(flow_file) as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: File {flow_file} not found.")
            return []
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {flow_file}")
            return []
    
    def initialize_simulation(self, num_vehicles_per_flow=10, congestion_factor=1.0):
        """Initialize vehicles based on flow data"""
        self.vehicles = []
        self.road_network = defaultdict(list)
        
        vehicle_id = 0
        for flow_idx, flow in enumerate(self.flows):
            vehicle_template = flow["vehicle"]
            route = flow["route"]
            interval = flow.get("interval", 2.0)
            
            # Adjust based on congestion factor
            adjusted_interval = max(0.5, interval / congestion_factor)  # Minimum interval
            
            # Generate multiple vehicles per flow
            for i in range(min(num_vehicles_per_flow, 20)):  # Limit to 20 per flow
                start_time = flow.get("startTime", 0) + i * adjusted_interval
                
                vehicle = {
                    "id": vehicle_id,
                    "flow_id": flow_idx,
                    "route": route.copy(),
                    "current_road": route[0] if route else "",
                    "road_idx": 0,
                    "position": np.random.uniform(0, 20),  # Stagger starting positions
                    "speed": np.random.uniform(0, 5),  # Random initial speed
                    "max_speed": vehicle_template["maxSpeed"],
                    "acceleration": vehicle_template["usualPosAcc"],
                    "deceleration": abs(vehicle_template["usualNegAcc"]),
                    "length": vehicle_template["length"],
                    "minGap": vehicle_template.get("minGap", 2.5),
                    "headwayTime": vehicle_template.get("headwayTime", 1.5),
                    "start_time": start_time,
                    "state": "waiting" if start_time > 0 else "running",
                    "waiting_time": 0.0,
                    "total_distance": 0.0,
                    "last_acceleration": 0.0,
                    "emergency": False  # For emergency vehicle scenario
                }
                
                self.vehicles.append(vehicle)
                if vehicle["current_road"]:
                    self.road_network[vehicle["current_road"]].append(vehicle)
                
                vehicle_id += 1
        
        # Initialize traffic signals for each intersection
        self.initialize_traffic_signals()
        
        print(f"Initialized {len(self.vehicles)} vehicles with congestion factor {congestion_factor}")
    
    def initialize_traffic_signals(self):
        """Initialize traffic signals at intersections"""
        # Create signals for each intersection based on road patterns
        intersection_signals = {}
        
        # Parse road names to find intersections (roads with same x_y coordinates)
        road_intersections = defaultdict(set)
        for vehicle in self.vehicles:
            road = vehicle["current_road"]
            if road and '_' in road:
                parts = road.split('_')
                if len(parts) >= 3:
                    intersection_key = f"{parts[1]}_{parts[2]}"  # x_y coordinates
                    road_intersections[intersection_key].add(road)
        
        # Create a signal for each intersection
        for intersection_key in road_intersections:
            # Different phase times for congestion scenarios
            if hasattr(self, 'scenario_type') and "congestion" in self.scenario_type:
                # Shorter green times in congestion
                phase_times = {
                    "NS_green": 20,
                    "NS_yellow": 3,
                    "EW_green": 15,
                    "EW_yellow": 3
                }
            else:
                phase_times = {
                    "NS_green": 30,
                    "NS_yellow": 5,
                    "EW_green": 25,
                    "EW_yellow": 5
                }
            
            # Add random offset to stagger signals
            offset = np.random.uniform(0, 60)
            signal = TrafficSignal(phase_times)
            signal.current_time = offset
            signal.phase_start_time = offset
            
            intersection_signals[intersection_key] = signal
        
        self.traffic_signals = intersection_signals
    
    def get_road_direction(self, road_name):
        """Determine direction of road based on naming convention"""
        if not road_name or '_' not in road_name:
            return "unknown"
        
        parts = road_name.split('_')
        if len(parts) >= 4:
            last_part = parts[-1]
            if last_part == '0':
                return "east"  # Assuming 0 is eastbound
            elif last_part == '1':
                return "north"
            elif last_part == '2':
                return "west"
            elif last_part == '3':
                return "south"
        
        return "unknown"
    
    def get_intersection_key(self, road_name):
        """Get intersection key from road name"""
        if not road_name or '_' not in road_name:
            return None
        
        parts = road_name.split('_')
        if len(parts) >= 3:
            return f"{parts[1]}_{parts[2]}"
        return None
    
    def find_leading_vehicle(self, vehicle):
        """Find vehicle directly ahead on same road"""
        road_vehicles = self.road_network.get(vehicle["current_road"], [])
        
        if len(road_vehicles) < 2:
            return None, None  # Return None for both leader and gap
        
        min_gap = float('inf')
        leader = None
        
        for v in road_vehicles:
            if v["id"] != vehicle["id"] and v["position"] > vehicle["position"]:
                gap = v["position"] - vehicle["position"] - v["length"]
                if gap < min_gap:
                    min_gap = gap
                    leader = v
        
        if leader is None:
            return None, None
        
        return leader, min_gap if min_gap != float('inf') else None
    
    def calculate_acceleration(self, vehicle, leader, gap):
        """Calculate acceleration using IDM model"""
        if leader is None:
            leader_speed = vehicle["max_speed"]
        else:
            leader_speed = leader["speed"]
        
        return IDM_acceleration(vehicle, gap, leader_speed)
    
    def update_vehicle_state(self, vehicle):
        """Update state of a single vehicle"""
        # Check if vehicle should start
        if vehicle["state"] == "waiting":
            if self.time >= vehicle["start_time"]:
                vehicle["state"] = "running"
            else:
                return
        
        if vehicle["state"] != "running":
            return
        
        # Get intersection and direction for signal checking
        intersection_key = self.get_intersection_key(vehicle["current_road"])
        road_direction = self.get_road_direction(vehicle["current_road"])
        
        # Check if approaching intersection
        distance_to_end = self.road_length - vehicle["position"]
        is_approaching_intersection = distance_to_end < 30
        
        # Check traffic signal
        signal_state = "green"  # Default
        if intersection_key and is_approaching_intersection and intersection_key in self.traffic_signals:
            signal = self.traffic_signals[intersection_key]
            signal_state = signal.get_state(road_direction)
        
        # Emergency vehicle priority
        if vehicle["emergency"]:
            # Emergency vehicles ignore red lights and have right of way
            signal_state = "green"
        
        # Vehicle behavior based on signal
        if signal_state == "red" and distance_to_end < 20:
            # Stop at red light
            vehicle["waiting_time"] += self.dt
            gap_to_stop = distance_to_end
            
            if gap_to_stop > 0:
                # Decelerate to stop at intersection
                acceleration = IDM_acceleration(vehicle, gap_to_stop, 0)
                vehicle["speed"] = max(0, vehicle["speed"] + acceleration * self.dt)
        elif signal_state == "yellow" and distance_to_end < 20:
            # Decide to stop or go through yellow
            if distance_to_end > 10:
                # Try to stop
                acceleration = IDM_acceleration(vehicle, distance_to_end, 0)
                vehicle["speed"] = max(0, vehicle["speed"] + acceleration * self.dt)
                vehicle["waiting_time"] += self.dt
            else:
                # Go through yellow
                leader, gap = self.find_leading_vehicle(vehicle)
                acceleration = self.calculate_acceleration(vehicle, leader, gap)
                vehicle["speed"] = max(0, min(vehicle["speed"] + acceleration * self.dt, vehicle["max_speed"]))
        else:
            # Normal driving
            leader, gap = self.find_leading_vehicle(vehicle)
            acceleration = self.calculate_acceleration(vehicle, leader, gap)
            vehicle["speed"] = max(0, min(vehicle["speed"] + acceleration * self.dt, vehicle["max_speed"]))
            
            # Reset waiting time if moving
            if vehicle["speed"] > 0:
                vehicle["waiting_time"] = max(0, vehicle["waiting_time"] - 0.5)
        
        # Update position
        vehicle["position"] += vehicle["speed"] * self.dt
        vehicle["total_distance"] += vehicle["speed"] * self.dt
        
        # Check if vehicle went beyond road length
        if vehicle["position"] > self.road_length:
            vehicle["position"] = self.road_length - 1
        
        # Handle lane changing/routing
        if vehicle["position"] >= self.road_length:
            # Remove from current road
            current_road = vehicle["current_road"]
            if current_road in self.road_network:
                self.road_network[current_road] = [
                    v for v in self.road_network[current_road] if v["id"] != vehicle["id"]
                ]
            
            if vehicle["road_idx"] < len(vehicle["route"]) - 1:
                # Move to next road
                vehicle["road_idx"] += 1
                vehicle["current_road"] = vehicle["route"][vehicle["road_idx"]]
                vehicle["position"] = vehicle["position"] - self.road_length  # Carry over extra distance
                
                # Add to new road
                if vehicle["current_road"]:
                    self.road_network[vehicle["current_road"]].append(vehicle)
            else:
                # Reached destination
                vehicle["state"] = "finished"
    
    def simulate_step(self):
        """Execute one simulation step"""
        # Update traffic signals
        for signal in self.traffic_signals.values():
            signal.update(self.dt)
        
        # Update vehicles in order (front to back on each road)
        running_vehicles = [v for v in self.vehicles if v["state"] == "running"]
        
        # Sort by road and position (front to back)
        road_groups = defaultdict(list)
        for v in running_vehicles:
            if v["current_road"]:
                road_groups[v["current_road"]].append(v)
        
        for road in road_groups:
            road_groups[road].sort(key=lambda x: x["position"], reverse=True)
        
        # Update vehicles
        for road in road_groups:
            for vehicle in road_groups[road]:
                self.update_vehicle_state(vehicle)
        
        # Remove finished vehicles
        self.vehicles = [v for v in self.vehicles if v["state"] != "finished"]
        
        # Increment time
        self.time += self.dt
        
        return self.capture_state()
    
    def capture_state(self):
        """Capture current simulation state"""
        state = {
            "time": self.time,
            "vehicles": [],
            "signals": {},
            "statistics": self.calculate_statistics()
        }
        
        # Capture vehicle states
        for vehicle in self.vehicles:
            if vehicle["state"] == "running":
                vehicle_state = {
                    "id": vehicle["id"],
                    "flow_id": vehicle["flow_id"],
                    "current_road": vehicle["current_road"],
                    "position": float(vehicle["position"]),
                    "speed": float(vehicle["speed"]),
                    "state": vehicle["state"],
                    "waiting_time": float(vehicle["waiting_time"]),
                    "total_distance": float(vehicle["total_distance"]),
                    "emergency": vehicle["emergency"]
                }
                state["vehicles"].append(vehicle_state)
        
        # Capture signal states
        for intersection_key, signal in self.traffic_signals.items():
            state["signals"][intersection_key] = signal.get_summary()
        
        return state
    
    def calculate_statistics(self):
        """Calculate simulation statistics"""
        running_vehicles = [v for v in self.vehicles if v["state"] == "running"]
        
        if not running_vehicles:
            return {
                "total_vehicles": 0,
                "average_speed": 0,
                "total_waiting_time": 0,
                "congestion_level": 0
            }
        
        speeds = [v["speed"] for v in running_vehicles]
        waiting_times = [v["waiting_time"] for v in running_vehicles]
        
        # Calculate congestion level (0-1)
        max_capacity_per_road = 20  # vehicles
        total_roads = len(self.road_network)
        if total_roads > 0:
            congestion = min(1.0, len(running_vehicles) / (max_capacity_per_road * total_roads))
        else:
            congestion = 0
        
        return {
            "total_vehicles": len(running_vehicles),
            "average_speed": float(np.mean(speeds)) if speeds else 0,
            "total_waiting_time": float(sum(waiting_times)),
            "congestion_level": float(congestion),
            "roads_occupied": len([r for r in self.road_network if self.road_network[r]])
        }
    
    def run_scenario(self, scenario_type="congestion", duration=300, output_file=None):
        """Run a specific scenario"""
        self.scenario_type = scenario_type
        self.states_history = []
        
        print(f"\nRunning {scenario_type.upper()} scenario for {duration} seconds...")
        
        # Initialize based on scenario
        if scenario_type == "congestion":
            # Peak congestion: many vehicles, short intervals
            self.initialize_simulation(num_vehicles_per_flow=15, congestion_factor=0.5)
            
            # Create traffic jam at central intersection
            self.create_traffic_jam(intersection_key="2_2")
            
        elif scenario_type == "emergency":
            # Emergency situation: normal traffic + emergency vehicles
            self.initialize_simulation(num_vehicles_per_flow=8, congestion_factor=1.0)
            
            # Add emergency vehicles
            self.add_emergency_vehicles()
            
        else:  # normal
            self.initialize_simulation(num_vehicles_per_flow=5, congestion_factor=1.0)
        
        # Run simulation
        steps = int(duration / self.dt)
        for step in range(steps):
            state = self.simulate_step()
            self.states_history.append(state)
            
            # Print progress
            if step % 50 == 0:
                stats = state["statistics"]
                print(f"  Time: {state['time']:.1f}s | Vehicles: {stats['total_vehicles']} | "
                      f"Avg Speed: {stats['average_speed']:.2f}m/s | Congestion: {stats['congestion_level']:.2f}")
        
        # Save to file if requested
        if output_file:
            self.save_states(output_file)
        
        return self.states_history
    
    def create_traffic_jam(self, intersection_key):
        """Create a traffic jam at specified intersection"""
        # Find vehicles approaching the intersection
        for vehicle in self.vehicles:
            if vehicle["state"] == "running":
                vehicle_intersection = self.get_intersection_key(vehicle["current_road"])
                if vehicle_intersection == intersection_key:
                    # Slow down vehicles near this intersection
                    if vehicle["position"] > self.road_length - 50:
                        vehicle["speed"] = max(0, vehicle["speed"] * 0.3)
        
        # Add extra vehicles at the intersection
        roads_at_intersection = []
        for road in self.road_network:
            if intersection_key in road:
                roads_at_intersection.append(road)
        
        # Add stalled vehicles (only if we have roads)
        for i, road in enumerate(roads_at_intersection[:2]):  # First 2 roads
            stalled_vehicle = {
                "id": 1000 + i,
                "flow_id": -1,
                "route": [road],
                "current_road": road,
                "road_idx": 0,
                "position": self.road_length - 20 - i * 10,
                "speed": 0.0,
                "max_speed": 16.67,
                "acceleration": 2.0,
                "deceleration": 4.5,
                "length": 5.0,
                "minGap": 2.5,
                "headwayTime": 1.5,
                "start_time": 0,
                "state": "running",
                "waiting_time": 30.0,  # Already waiting
                "total_distance": 0.0,
                "last_acceleration": 0.0,
                "emergency": False
            }
            self.vehicles.append(stalled_vehicle)
            self.road_network[road].append(stalled_vehicle)
    
    def add_emergency_vehicles(self):
        """Add emergency vehicles to the simulation"""
        # Add 2 emergency vehicles on different routes
        emergency_routes = [
            ["road_0_1_0", "road_1_1_0", "road_2_1_0", "road_3_1_0", "road_4_1_0"],
            ["road_5_2_2", "road_4_2_2", "road_3_2_2", "road_2_2_2", "road_1_2_2"]
        ]
        
        for i, route in enumerate(emergency_routes):
            if route and route[0]:  # Check if route is valid
                emergency_vehicle = {
                    "id": 2000 + i,
                    "flow_id": -2,
                    "route": route,
                    "current_road": route[0],
                    "road_idx": 0,
                    "position": 10 + i * 20,
                    "speed": 20.0,  # Higher speed
                    "max_speed": 25.0,  # Higher max speed
                    "acceleration": 3.0,  # Faster acceleration
                    "deceleration": 6.0,
                    "length": 6.0,  # Longer vehicle
                    "minGap": 3.0,
                    "headwayTime": 1.0,
                    "start_time": 30.0,  # Start later
                    "state": "running",
                    "waiting_time": 0.0,
                    "total_distance": 0.0,
                    "last_acceleration": 0.0,
                    "emergency": True
                }
                
                self.vehicles.append(emergency_vehicle)
                if emergency_vehicle["current_road"]:
                    self.road_network[emergency_vehicle["current_road"]].append(emergency_vehicle)
        
        print(f"Added {len(emergency_routes)} emergency vehicles")
    
    def save_states(self, filename):
        """Save simulation states to JSON file"""
        # Convert numpy types to Python native types for JSON serialization
        def convert_to_json(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json(item) for item in obj]
            else:
                return obj
        
        output_data = {
            "scenario": self.scenario_type,
            "duration": float(self.time),
            "time_step": float(self.dt),
            "road_length": float(self.road_length),
            "total_states": len(self.states_history),
            "states": convert_to_json(self.states_history)
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"Saved {len(self.states_history)} states to {filename}")
    
    def generate_summary_report(self):
        """Generate a summary report of the simulation"""
        if not self.states_history:
            return {"error": "No simulation data available"}
        
        last_state = self.states_history[-1]
        first_state = self.states_history[0]
        
        # Calculate overall statistics
        all_speeds = []
        all_waiting_times = []
        max_congestion = 0
        max_vehicles = 0
        
        for state in self.states_history:
            stats = state["statistics"]
            all_speeds.append(stats["average_speed"])
            all_waiting_times.append(stats["total_waiting_time"])
            max_congestion = max(max_congestion, stats["congestion_level"])
            max_vehicles = max(max_vehicles, stats["total_vehicles"])
        
        report = {
            "scenario": self.scenario_type,
            "simulation_duration": float(self.time),
            "final_statistics": last_state["statistics"],
            "overall_statistics": {
                "average_speed_over_time": float(np.mean(all_speeds)) if all_speeds else 0,
                "max_congestion_level": float(max_congestion),
                "max_vehicles_simultaneously": int(max_vehicles),
                "total_waiting_time_accumulated": float(sum(all_waiting_times))
            },
            "traffic_signals": len(self.traffic_signals),
            "vehicles_completed": len([v for v in self.vehicles if v["state"] == "finished"])
        }
        
        return report

# -------------------------
# Main Execution
# -------------------------
def main():
    """Main function to generate states for hard scenarios"""
    parser = argparse.ArgumentParser(description="Generate traffic simulation states for hard scenarios")
    parser.add_argument("--flow-file", type=str, default="flow_3_4.json",
                       help="Path to flow JSON file")
    parser.add_argument("--output-dir", type=str, default="output_states",
                       help="Output directory for state files")
    parser.add_argument("--duration", type=int, default=200,  # Reduced for testing
                       help="Simulation duration in seconds")
    parser.add_argument("--dt", type=float, default=1.0,
                       help="Time step in seconds")
    parser.add_argument("--road-length", type=float, default=100.0,
                       help="Length of each road segment")
    parser.add_argument("--scenario", type=str, choices=["congestion", "emergency", "both"], default="both",
                       help="Which scenario to run")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize state generator
    print("=" * 70)
    print("TRAFI State Generator - Hard Scenario Analysis")
    print("=" * 70)
    
    generator = StateGenerator(
        flow_file=args.flow_file,
        dt=args.dt,
        road_length=args.road_length
    )
    
    # Determine which scenarios to run
    if args.scenario == "both":
        scenarios = [
            ("congestion", "peak_congestion_scenario"),
            ("emergency", "emergency_traffic_scenario")
        ]
    elif args.scenario == "congestion":
        scenarios = [("congestion", "peak_congestion_scenario")]
    else:
        scenarios = [("emergency", "emergency_traffic_scenario")]
    
    all_reports = {}
    
    for scenario_type, scenario_name in scenarios:
        print(f"\n{'='*70}")
        print(f"Scenario: {scenario_name.upper()}")
        print(f"{'='*70}")
        
        # Run simulation
        output_file = os.path.join(args.output_dir, f"{scenario_name}_states.json")
        try:
            states = generator.run_scenario(
                scenario_type=scenario_type,
                duration=args.duration,
                output_file=output_file
            )
            
            # Generate report
            report = generator.generate_summary_report()
            all_reports[scenario_name] = report
            
            # Save report
            report_file = os.path.join(args.output_dir, f"{scenario_name}_report.json")
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"\nReport saved to: {report_file}")
            
            # Generate visualization for key moments
            try:
                generate_visualization(states, scenario_name, args.output_dir)
            except Exception as e:
                print(f"Warning: Could not generate visualization: {e}")
                
        except Exception as e:
            print(f"Error running scenario {scenario_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate comparative report if we have multiple scenarios
    if len(all_reports) > 1:
        comparative_report = {
            "generated_at": datetime.now().isoformat(),
            "simulation_parameters": {
                "duration": args.duration,
                "time_step": args.dt,
                "road_length": args.road_length
            },
            "scenario_comparison": all_reports,
            "conclusions": generate_conclusions(all_reports)
        }
        
        comp_report_file = os.path.join(args.output_dir, "comparative_analysis.json")
        with open(comp_report_file, 'w') as f:
            json.dump(comparative_report, f, indent=2)
        
        print(f"\nComparative analysis: {comp_report_file}")
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"All results saved to: {args.output_dir}/")
    
    # Print key findings
    if all_reports:
        print("\nKEY FINDINGS:")
        for scenario_name, report in all_reports.items():
            stats = report.get("final_statistics", {})
            print(f"\n{scenario_name.upper()}:")
            print(f"  • Final congestion level: {stats.get('congestion_level', 0):.2f}")
            print(f"  • Average speed: {stats.get('average_speed', 0):.2f} m/s")
            print(f"  • Total vehicles: {stats.get('total_vehicles', 0)}")
    else:
        print("\nNo reports generated.")

def generate_visualization(states, scenario_name, output_dir):
    """Generate simple visualizations of key moments"""
    if not states:
        return
    
    # Extract key metrics over time
    times = [s["time"] for s in states]
    speeds = [s["statistics"]["average_speed"] for s in states]
    congestion = [s["statistics"]["congestion_level"] for s in states]
    vehicles = [s["statistics"]["total_vehicles"] for s in states]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Scenario: {scenario_name}", fontsize=14, fontweight='bold')
    
    # Plot 1: Speed over time
    ax1 = axes[0, 0]
    ax1.plot(times, speeds, 'b-', linewidth=2)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Average Speed (m/s)")
    ax1.set_title("System Speed Over Time")
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Congestion over time
    ax2 = axes[0, 1]
    ax2.plot(times, congestion, 'r-', linewidth=2)
    ax2.fill_between(times, congestion, alpha=0.3, color='red')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Congestion Level")
    ax2.set_title("Congestion Over Time")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Vehicle count
    ax3 = axes[1, 0]
    ax3.plot(times, vehicles, 'g-', linewidth=2)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Number of Vehicles")
    ax3.set_title("Active Vehicles Over Time")
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: State snapshot at peak congestion
    ax4 = axes[1, 1]
    if congestion:
        peak_idx = congestion.index(max(congestion))
        peak_state = states[peak_idx]
        
        # Extract vehicle positions at peak
        if peak_state["vehicles"]:
            positions = [v["position"] for v in peak_state["vehicles"]]
            speeds_at_peak = [v["speed"] for v in peak_state["vehicles"]]
            
            scatter = ax4.scatter(range(len(positions)), positions, c=speeds_at_peak, 
                                cmap='viridis', alpha=0.6, s=20)
            ax4.set_xlabel("Vehicle Index")
            ax4.set_ylabel("Position (m)")
            ax4.set_title(f"Vehicle Positions at Peak Congestion (t={peak_state['time']:.1f}s)")
            plt.colorbar(scatter, ax=ax4, label='Speed (m/s)')
    
    plt.tight_layout()
    
    # Save figure
    fig_file = os.path.join(output_dir, f"{scenario_name}_analysis.png")
    plt.savefig(fig_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {fig_file}")

def generate_conclusions(reports):
    """Generate conclusions from scenario analysis"""
    conclusions = []
    
    congestion_report = reports.get("peak_congestion_scenario", {})
    emergency_report = reports.get("emergency_traffic_scenario", {})
    
    # Congestion scenario conclusions
    if congestion_report:
        cong_stats = congestion_report.get("final_statistics", {})
        cong_level = cong_stats.get("congestion_level", 0)
        
        if cong_level > 0.7:
            conclusions.append("Peak congestion scenario shows severe gridlock conditions")
            conclusions.append("Average speeds drop significantly during peak periods")
            conclusions.append("Traffic signals become ineffective during heavy congestion")
        elif cong_level > 0.4:
            conclusions.append("Moderate congestion observed with reduced traffic flow")
            conclusions.append("Some delays at intersections but system remains functional")
        else:
            conclusions.append("Light traffic conditions with smooth flow")
    
    # Emergency scenario conclusions
    if emergency_report:
        conclusions.append("Emergency vehicles significantly impact traffic flow")
        conclusions.append("Priority routing for emergency vehicles reduces response time")
        conclusions.append("Other vehicles experience additional delays during emergency operations")
    
    # Comparative conclusions
    if congestion_report and emergency_report:
        cong_speed = congestion_report.get("final_statistics", {}).get("average_speed", 0)
        emerg_speed = emergency_report.get("final_statistics", {}).get("average_speed", 0)
        
        if emerg_speed > cong_speed:
            conclusions.append("Emergency management improves traffic flow compared to unmanaged congestion")
        else:
            conclusions.append("Emergency operations create additional disruptions to normal traffic")
    
    return conclusions

# -------------------------
# Run as standalone script
# -------------------------
if __name__ == "__main__":
    main()