# TRAFI - Traffic Flow Simulator 
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import argparse
import os
import sys
from datetime import datetime
import math 

INTERSECTIONS = {
    0: ["road_0", "road_1"],
    1: ["road_2", "road_3"],
    2: ["road_4", "road_5"]
}


# -------------------------
# Car-Following Models
# -------------------------
def IDM_acceleration(vehicle, gap, leader_speed, min_gap=2.0, desired_headway=1.5):
    """
    Intelligent Driver Model (IDM)
    Returns acceleration based on current conditions
    """
    v = vehicle["speed"]
    v0 = vehicle["max_speed"]
    s0 = min_gap
    T = desired_headway
    a = vehicle["acceleration"]
    b = abs(vehicle["deceleration"])
    
    # Desired minimum gap
    delta_v = v - leader_speed
    s_star = s0 + max(0, v * T + (v * delta_v) / (2 * np.sqrt(a * b)))
    
    # IDM acceleration formula
    if gap > 0:
        acceleration = a * (1 - (v / v0)**4 - (s_star / gap)**2)
    else:
        acceleration = -b  # Emergency braking
    
    return max(min(acceleration, a), -b)

import numpy as np

def Gipps_acceleration(vehicle, gap, leader_speed, leader_deceleration=3.0, reaction_time=1.0):
    """
    Gipps' Car-Following Model
    More conservative than IDM
    Robust version with safe defaults
    """
    # --- Safe vehicle parameters ---
    v = vehicle.get("speed", 0.0)
    v0 = vehicle.get("max_speed", 30.0)  # default max speed if missing
    a = vehicle.get("acceleration", 2.0)  # default acceleration
    b = abs(vehicle.get("deceleration", 4.5))  # safe deceleration

    # --- Safe leader parameters ---
    if leader_speed is None:
        leader_speed = 0.0
    if leader_deceleration is None or leader_deceleration <= 0:
        leader_deceleration = 3.0
    if gap is None or gap < 0:
        gap = 999.0  # very large gap if no leader

    # --- Term1: free-road acceleration ---
    term1 = 2.5 * a * reaction_time * (1 - v / v0) * np.sqrt(0.025 + v / v0)

    # --- Term2: safe distance braking ---
    sqrt_arg = b**2 * reaction_time**2 + b * (2 * gap - v * reaction_time - leader_speed**2 / leader_deceleration)
    if sqrt_arg < 0:
        sqrt_arg = 0.0  # prevent invalid sqrt
    term2 = b * reaction_time + np.sqrt(sqrt_arg)

    # --- Return acceleration, cannot exceed term2 - current speed ---
    acceleration = min(term1, term2 - v)

    return acceleration



def safe_acceleration(vehicle, leader, gap):
    """Safe car-following model (simplified)"""
    if leader is None:
        desired_acc = (vehicle["max_speed"] - vehicle["speed"]) / 2.0
        return min(desired_acc, vehicle["acceleration"])
    
    safe_distance = vehicle["speed"] * 1.0 + vehicle["length"] * 1.5
    if gap < safe_distance:
        required_decel = (vehicle["speed"]**2 - leader["speed"]**2) / (2 * max(gap, 0.1))
        return -min(abs(required_decel), vehicle["deceleration"])
    else:
        return vehicle["acceleration"]

# -------------------------
# Traffic Signal Class
# -------------------------
class TrafficSignal:
    def __init__(self, cycle_time=60, green_time=30, offset=0):
        self.cycle_time = cycle_time
        self.green_time = green_time
        self.offset = offset

        self.current_time = 0.0
        self.phase_history = []

        # ===============================
        # RL-CONTROLLED STATE (NEW)
        # ===============================
        self.current_phase = 0      # 0 = even roads green, 1 = odd roads green
        self.last_switch_time = 0.0
        self.min_green = 5.0        # seconds (safety constraint)

    def get_phase(self, road_id):
        """
        Get signal phase for given road
        RL-controlled (overrides fixed cycle)
        """
        if road_id % 2 == 0:
            return "green" if self.current_phase == 0 else "red"
        else:
            return "green" if self.current_phase == 1 else "red"

    def is_green(self, road_id):
        return self.get_phase(road_id) == "green"

    def update(self, dt):
        """Update signal timing"""
        self.current_time += dt
        self.phase_history.append(self.current_phase)

    # ===============================
    # 🔥 RL ACTION INTERFACE (REQUIRED)
    # ===============================
    def apply_action(self, action, sim_time):
        """
        RL action:
        action = 0 → even roads green
        action = 1 → odd roads green
        """
        # Enforce minimum green time
        if sim_time - self.last_switch_time < self.min_green:
            return

        if action != self.current_phase:
            self.current_phase = action
            self.last_switch_time = sim_time

    def get_schedule(self):
        """Get signal schedule information"""
        return {
            "current_phase": self.current_phase,
            "last_switch_time": self.last_switch_time,
            "min_green": self.min_green
        }



class RLTrafficSignal(TrafficSignal):
    """
    RL-controlled traffic signal
    Actions:
        0 -> Even roads green
        1 -> Odd roads green
    """

    def __init__(self, cycle_time=60, min_green=10):
        super().__init__(cycle_time=cycle_time, green_time=cycle_time//2)
        self.min_green = min_green
        self.last_switch_time = 0
        self.current_action = 0

    def apply_action(self, action, current_time):
        if action != self.current_action:
            if current_time - self.last_switch_time >= self.min_green:
                self.current_action = action
                self.last_switch_time = current_time

    def get_phase(self, road_id):
        if self.current_action == 0:
            return "green" if road_id % 2 == 0 else "red"
        else:
            return "green" if road_id % 2 == 1 else "red"


# -------------------------
# Road Network Class
# -------------------------
class RoadNetwork:
    def __init__(self, road_length=100):
        self.road_length = road_length
        self.roads = defaultdict(list)  # road_id -> list of vehicles
        
    def add_vehicle(self, road_id, vehicle):
        """Add vehicle to road"""
        self.roads[road_id].append(vehicle)
        
    def remove_vehicle(self, road_id, vehicle_id):
        """Remove vehicle from road"""
        self.roads[road_id] = [v for v in self.roads[road_id] if v["id"] != vehicle_id]
        
    def get_vehicles_on_road(self, road_id):
        """Get all vehicles on specific road"""
        return sorted(self.roads[road_id], key=lambda x: x["position"])
    
    def get_road_density(self, road_id):
        """Calculate density on a road (vehicles per meter)"""
        vehicles = self.roads[road_id]
        return len(vehicles) / self.road_length if vehicles else 0
    
    def get_total_vehicles(self):
        """Get total number of vehicles in network"""
        return sum(len(vehicles) for vehicles in self.roads.values())

# -------------------------
# Statistics Collection
# -------------------------
class TRAFI_Stats:
    def __init__(self):
        self.travel_times = []
        self.average_speeds = []
        self.delays = []
        self.waiting_times = []
        self.fuel_consumption = []
        self.emissions = []
        self.road_occupancy = defaultdict(list)
        self.time_series_data = {
            "speed": [],
            "density": [],
            "flow": [],
            "queue_length": [],
            "total_vehicles": []
        }
        self.start_time = datetime.now()
        
    def calculate_fuel_consumption(self, vehicle, time_step):
        """Estimate fuel consumption based on speed and acceleration"""
        # Simplified fuel model (liters per second)
        base_rate = 0.0000556  # 0.2 liters per hour at idle
        speed_factor = vehicle["speed"] * 0.0000139
        acc_factor = abs(vehicle.get("last_acceleration", 0)) * 0.0000278
        
        fuel = (base_rate + speed_factor + acc_factor) * time_step
        self.fuel_consumption.append(fuel)
        return fuel
    
    def calculate_emissions(self, vehicle, time_step):
        """Estimate CO2 emissions (grams per second)"""
        # Simplified emission model
        if vehicle["speed"] == 0:
            emission_rate = 0.2778  # 1 gram per 3.6 seconds at idle
        elif vehicle["speed"] < 5:
            emission_rate = 1.3889  # 5 grams per 3.6 seconds when crawling
        elif vehicle["speed"] < 15:
            emission_rate = 0.8333  # 3 grams per 3.6 seconds in urban driving
        else:
            emission_rate = 0.5556  # 2 grams per 3.6 seconds on highway
            
        emissions = emission_rate * time_step
        self.emissions.append(emissions)
        return emissions
    
    def record_trip(self, vehicle, start_time, end_time, distance):
        """Record completed trip statistics"""
        travel_time = end_time - start_time
        self.travel_times.append(travel_time)
        
        avg_speed = distance / travel_time if travel_time > 0 else 0
        self.average_speeds.append(avg_speed)
        
        # Theoretical minimum time (free flow)
        min_time = distance / vehicle["max_speed"]
        delay = max(0, travel_time - min_time)
        self.delays.append(delay)
        
        if "waiting_time" in vehicle:
            self.waiting_times.append(vehicle["waiting_time"])
    
    def record_time_series(self, step, vehicles, road_network):
        """Record time-series data for fundamental diagram"""
        if vehicles:
            avg_speed = np.mean([v["speed"] for v in vehicles]) if vehicles else 0
            total_vehicles = road_network.get_total_vehicles()
            
            # Calculate flow based on completed trips in last time step
            if step > 0:
                flow_rate = len([v for v in vehicles if v.get("completed_step", 0) == step]) / 1.0  # vehicles per second
            else:
                flow_rate = 0
            
            self.time_series_data["speed"].append(avg_speed)
            self.time_series_data["total_vehicles"].append(total_vehicles)
            self.time_series_data["flow"].append(flow_rate * 3600)  # Convert to vehicles per hour
            
            # Count vehicles waiting at signals
            queue_length = sum(1 for v in vehicles if v["speed"] == 0 and v.get("waiting_time", 0) > 0)
            self.time_series_data["queue_length"].append(queue_length)
            
            # Calculate density (vehicles per km) - assuming average road length of 0.1km
            density = total_vehicles / (len(road_network.roads) * 0.1) if road_network.roads else 0
            self.time_series_data["density"].append(density)
    
    def print_summary(self):
        """Print comprehensive simulation statistics"""
        print("\n" + "=" * 70)
        print("TRAFI - TRAFFIC FLOW SIMULATION RESULTS")
        print("=" * 70)
        
        if self.travel_times:
            print("\n📊 PERFORMANCE METRICS")
            print("-" * 40)
            print(f"  • Average travel time: {np.mean(self.travel_times):.2f} s")
            print(f"  • Average speed: {np.mean(self.average_speeds):.2f} m/s ({np.mean(self.average_speeds)*3.6:.1f} km/h)")
            print(f"  • Average delay: {np.mean(self.delays):.2f} s")
            
            if self.waiting_times:
                print(f"  • Average waiting time: {np.mean(self.waiting_times):.2f} s")
            
            print(f"\n📈 THROUGHPUT & EFFICIENCY")
            print("-" * 40)
            print(f"  • Total trips completed: {len(self.travel_times)}")
            avg_trip_distance = np.mean([s*3.6 for s in self.average_speeds]) * np.mean(self.travel_times) / 3600
            print(f"  • Average trip distance: {avg_trip_distance:.2f} km")
            
            if self.travel_times:
                efficiency = (1 - np.mean(self.delays) / np.mean(self.travel_times)) * 100
                print(f"  • System efficiency: {efficiency:.1f}%")
            
            if self.fuel_consumption:
                print(f"\n🌱 ENVIRONMENTAL IMPACT")
                print("-" * 40)
                total_fuel = sum(self.fuel_consumption)
                total_co2 = sum(self.emissions) / 1000  # Convert to kg
                print(f"  • Total fuel consumption: {total_fuel:.2f} liters")
                print(f"  • Total CO2 emissions: {total_co2:.2f} kg")
                print(f"  • Average fuel per trip: {total_fuel/len(self.travel_times):.3f} liters")
                print(f"  • Average CO2 per trip: {total_co2/len(self.travel_times):.3f} kg")
            
            print(f"\n⏱️  SIMULATION STATISTICS")
            print("-" * 40)
            sim_duration = (datetime.now() - self.start_time).total_seconds()
            print(f"  • Simulation runtime: {sim_duration:.2f} seconds")
            print(f"  • Data points collected: {sum(len(lst) for lst in self.time_series_data.values())}")
        else:
            print("⚠️  No trips completed during simulation")
        
        print("=" * 70)

# -------------------------
# Main Simulation Class
# -------------------------
INTERSECTIONS = {
    "int_1": ["road_1", "road_2"],
    "int_2": ["road_3", "road_4"]
}

class TRAFI_Simulator:

    def __init__(self, flow_file=None, dt=1.0, total_steps=300, road_length=100, 
                 car_following_model="IDM", enable_signals=True):
        self.dt = dt
        self.total_steps = total_steps
        self.road_length = road_length
        self.car_following_model = car_following_model
        self.enable_signals = enable_signals
        
        self.vehicles = []
        self.finished_vehicles = []
        self.stats = TRAFI_Stats()
        self.road_network = RoadNetwork(road_length)
        
        # -----------------------------
        # MULTI-INTERSECTION SIGNALS
        # -----------------------------
        self.traffic_signals = {}
        if enable_signals:
            for iid in INTERSECTIONS:
                self.traffic_signals[iid] = TrafficSignal()
        
        # Load traffic flow
        if flow_file:
            self.load_flow_data(flow_file)
        else:
            print("⚠️  No flow file provided, using sample data")
            self.create_sample_data()
        
        # Initialize vehicles
        self.initialize_vehicles()
        
        print(f"\n🚗 TRAFI Simulator Initialized")
        print(f"   • Vehicles: {len(self.vehicles)}")
        print(f"   • Time step: {dt}s, Total steps: {total_steps}")
        print(f"   • Road length: {road_length}m")
        print(f"   • Car-following model: {car_following_model}")
        print(f"   • Traffic signals: {'Enabled' if enable_signals else 'Disabled'}")

    # Helper to find intersection for a road
    def get_intersection_id(road):
        """
        Returns the intersection ID for a given road
        """
        # Example implementation
        for iid, roads in INTERSECTIONS.items():
            if road in roads:
                return iid
        return None  # if road not found

        # -----------------------------
    
    # Vehicle update method (MULTI-INTERSECTION READY)
    # -----------------------------
    def update_vehicle_state(self, vehicle, step):
        if vehicle["state"] == "waiting":
            if step >= vehicle["start_time"]:
                vehicle["state"] = "running"
                vehicle["speed"] = 0.1
            else:
                return
        
        if vehicle["state"] != "running":
            return

        # Get road ID
        road_id = 0
        if vehicle["current_road"] and "_" in vehicle["current_road"]:
            try:
                road_id = int(vehicle["current_road"].split("_")[-1])
            except:
                pass

        # -----------------------------
        # MULTI-INTERSECTION SIGNAL CHECK
        # -----------------------------
        signal_phase = "green"
        if self.enable_signals:
            iid = self.get_intersection_id(vehicle["current_road"])
            if iid and iid in self.traffic_signals:
                signal = self.traffic_signals[iid]
                signal_phase = signal.get_phase(road_id)

        # SIGNAL-AWARE DRIVING
        if signal_phase == "red" and vehicle["position"] > self.road_length - 30:
            vehicle["waiting_time"] += self.dt
            gap_to_stop = self.road_length - vehicle["position"]
            if gap_to_stop > 0:
                acceleration = IDM_acceleration(vehicle, gap_to_stop, 0)
                vehicle["speed"] = max(0, vehicle["speed"] + acceleration * self.dt)
        else:
            leader, gap = self.find_leading_vehicle(vehicle)
            acceleration = self.calculate_acceleration(vehicle, leader, gap)
            vehicle["last_acceleration"] = acceleration
            vehicle["speed"] = max(0, min(vehicle["speed"] + acceleration * self.dt, vehicle["max_speed"]))
            if vehicle["speed"] > 0.1:
                vehicle["waiting_time"] = max(0, vehicle["waiting_time"] - 0.5)

        # Update position
        delta = vehicle["speed"] * self.dt
        vehicle["position"] += delta
        vehicle["total_distance"] += delta

        # Environmental impact
        self.stats.calculate_fuel_consumption(vehicle, self.dt)
        self.stats.calculate_emissions(vehicle, self.dt)

        # Record trajectory
        vehicle["trajectory"].append({
            "time": step * self.dt,
            "road": vehicle["current_road"],
            "position": vehicle["position"],
            "speed": vehicle["speed"],
            "acceleration": vehicle.get("last_acceleration", 0)
        })

        # Road transition
        if vehicle["position"] >= self.road_length:
            self.road_network.remove_vehicle(vehicle["current_road"], vehicle["id"])
            if vehicle["road_idx"] < len(vehicle["route"]) - 1:
                vehicle["road_idx"] += 1
                vehicle["current_road"] = vehicle["route"][vehicle["road_idx"]]
                vehicle["position"] = 0
                self.road_network.add_vehicle(vehicle["current_road"], vehicle)
            else:
                vehicle["state"] = "finished"
                vehicle["completed_step"] = step
                self.stats.record_trip(
                    vehicle,
                    vehicle["start_time"],
                    step * self.dt,
                    len(vehicle["route"]) * self.road_length
                )
                self.finished_vehicles.append(vehicle)  


    def load_flow_data(self, flow_file):
        """Load vehicle flow data from JSON file"""
        try:
            with open(flow_file) as f:
                self.flows = json.load(f)
            print(f"✅ Loaded flow data from {flow_file}")
        except FileNotFoundError:
            print(f"❌ Error: File {flow_file} not found.")
            print("   Using sample data instead.")
            self.create_sample_data()
        except json.JSONDecodeError:
            print(f"❌ Error: Invalid JSON format in {flow_file}")
            print("   Using sample data instead.")
            self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample flow data for testing"""
        print("📝 Generating sample traffic flow data...")
        sample_flows = []
        num_vehicles = 30
        
        for i in range(num_vehicles):
            # Randomize vehicle parameters
            max_speed = np.random.uniform(13.89, 25)  # 50-90 km/h
            length = np.random.uniform(4.0, 5.0)
            acc = np.random.uniform(1.5, 3.0)
            dec = np.random.uniform(3.0, 6.0)
            
            # Create route with 2-4 road segments
            num_roads = np.random.randint(2, 5)
            route = [f"road_{j}" for j in range(num_roads)]
            
            sample_flows.append({
                "vehicle": {
                    "length": length,
                    "maxSpeed": max_speed,
                    "usualPosAcc": acc,
                    "usualNegAcc": -dec
                },
                "route": route,
                "startTime": i * np.random.uniform(2, 6)  # Stagger start times
            })
        
        self.flows = sample_flows
        print(f"✅ Generated {len(sample_flows)} sample vehicles")
    
    def initialize_vehicles(self):
        """Initialize vehicles from flow data with safe spacing"""
        road_positions = defaultdict(float)

        for idx, flow in enumerate(self.flows):
            vehicle_info = flow["vehicle"]
            route = flow["route"]
            start_time = flow.get("startTime", 0)

            # Safe staggered start positions
            base_gap = 8.0  # meters
            start_pos = road_positions[route[0]] if route else 0.0
            road_positions[route[0]] += base_gap

            vehicle = {
                "id": idx,
                "route": route,
                "current_road": route[0] if route else "",
                "road_idx": 0,
                "position": start_pos,
                "speed": np.random.uniform(2.0, 5.0),  # initial rolling speed
                "max_speed": vehicle_info["maxSpeed"],
                "acceleration": vehicle_info["usualPosAcc"],
                "deceleration": abs(vehicle_info["usualNegAcc"]),
                "length": vehicle_info["length"],
                "start_time": start_time,
                "state": "running",  # START MOVING IMMEDIATELY
                "trajectory": [],
                "waiting_time": 0.0,
                "total_distance": 0.0,
                "last_acceleration": 0.0,
                "completed_step": None
            }

            self.vehicles.append(vehicle)
            self.road_network.add_vehicle(vehicle["current_road"], vehicle) 

    def find_leading_vehicle(self, vehicle):
        """Find the vehicle directly ahead on the same road"""
        road_vehicles = self.road_network.get_vehicles_on_road(vehicle["current_road"])
        
        if len(road_vehicles) < 2:
            return None, float('inf')
        
        # Find vehicle with smallest positive position difference
        min_gap = float('inf')
        leader = None
        
        for v in road_vehicles:
            if v["id"] != vehicle["id"] and v["position"] > vehicle["position"]:
                gap = v["position"] - vehicle["position"] - v["length"]
                if gap < min_gap:
                    min_gap = gap
                    leader = v
        
        return leader, min_gap if min_gap != float('inf') else None
    
    def calculate_acceleration(self, vehicle, leader, gap):
        """Calculate acceleration based on selected car-following model"""
        leader_speed = leader["speed"] if leader else vehicle["max_speed"]
        
        if self.car_following_model.upper() == "IDM":
            acceleration = IDM_acceleration(vehicle, gap, leader_speed)
        elif self.car_following_model.upper() == "GIPPS":
            acceleration = Gipps_acceleration(vehicle, gap, leader_speed)
        else:  # Default to safe model
            acceleration = safe_acceleration(vehicle, leader, gap)
        
        vehicle["last_acceleration"] = acceleration
        return acceleration
    
    def update_vehicle_state(self, vehicle, step):
        """Update state of a single vehicle (MULTI-INTERSECTION READY)"""

        # -----------------------------
        # Vehicle start logic
        # -----------------------------
        if vehicle["state"] == "waiting":
            if step >= vehicle["start_time"]:
                vehicle["state"] = "running"
                vehicle["speed"] = 0.1
            else:
                return

        if vehicle["state"] != "running":
            return

        # -----------------------------
        # Road ID (for signal phase)
        # -----------------------------
        road_id = 0
        if vehicle["current_road"] and "_" in vehicle["current_road"]:
            try:
                road_id = int(vehicle["current_road"].split("_")[-1])
            except:
                pass

        # -----------------------------
        # MULTI-INTERSECTION SIGNAL CHECK
        # -----------------------------
        signal_phase = "green"

        if self.enable_signals:
            iid = self.get_intersection_id(vehicle["current_road"])
            if iid is not None:
                signal = self.traffic_signals[iid]
                signal_phase = signal.get_phase(road_id)

        # -----------------------------
        # SIGNAL AWARE DRIVING
        # -----------------------------
        if signal_phase == "red" and vehicle["position"] > self.road_length - 30:

            vehicle["waiting_time"] += self.dt
            gap_to_stop = self.road_length - vehicle["position"]

            if gap_to_stop > 0:
                acceleration = IDM_acceleration(vehicle, gap_to_stop, 0)
                vehicle["speed"] = max(0, vehicle["speed"] + acceleration * self.dt)

        else:
            leader, gap = self.find_leading_vehicle(vehicle)
            acceleration = self.calculate_acceleration(vehicle, leader, gap)

            vehicle["last_acceleration"] = acceleration

            new_speed = vehicle["speed"] + acceleration * self.dt
            vehicle["speed"] = max(0, min(new_speed, vehicle["max_speed"]))

            if vehicle["speed"] > 0.1:
                vehicle["waiting_time"] = max(0, vehicle["waiting_time"] - 0.5)

        # -----------------------------
        # POSITION UPDATE
        # -----------------------------
        delta = vehicle["speed"] * self.dt
        vehicle["position"] += delta
        vehicle["total_distance"] += delta

        # -----------------------------
        # ENVIRONMENTAL METRICS
        # -----------------------------
        self.stats.calculate_fuel_consumption(vehicle, self.dt)
        self.stats.calculate_emissions(vehicle, self.dt)

        # -----------------------------
        # TRAJECTORY LOGGING
        # -----------------------------
        vehicle["trajectory"].append({
            "time": step * self.dt,
            "road": vehicle["current_road"],
            "position": vehicle["position"],
            "speed": vehicle["speed"],
            "acceleration": vehicle.get("last_acceleration", 0)
        })

        # -----------------------------
        # ROAD TRANSITION
        # -----------------------------
        if vehicle["position"] >= self.road_length:

            self.road_network.remove_vehicle(vehicle["current_road"], vehicle["id"])

            if vehicle["road_idx"] < len(vehicle["route"]) - 1:
                vehicle["road_idx"] += 1
                vehicle["current_road"] = vehicle["route"][vehicle["road_idx"]]
                vehicle["position"] = 0
                self.road_network.add_vehicle(vehicle["current_road"], vehicle)

            else:
                vehicle["state"] = "finished"
                vehicle["completed_step"] = step

                self.stats.record_trip(
                    vehicle,
                    vehicle["start_time"],
                    step * self.dt,
                    len(vehicle["route"]) * self.road_length
                )

                self.finished_vehicles.append(vehicle)

    def run_simulation_step(self, step):
        """
        Run a single simulation step:
        - Update all traffic signals
        - Update all vehicles
        - Record statistics
        """
        # ----------------------------
        # Update all traffic signals
        # ----------------------------
        if self.enable_signals:
            for signal in self.traffic_signals.values():
                signal.update(self.dt)

        # ----------------------------
        # Update all vehicles
        # ----------------------------
        for vehicle in self.vehicles:
            if vehicle["state"] != "finished":
                self.update_vehicle_state(vehicle, step)

        # ----------------------------
        # Collect statistics or debug info if needed
        # ----------------------------
        # Example: average speed
        speeds = [v["speed"] for v in self.vehicles if v["state"] == "running"]
        avg_speed = sum(speeds)/len(speeds) if speeds else 0
        # Optional: print every 50 steps
        if step % 50 == 0:
            print(f"Step {step}: Running vehicles: {len(speeds)}, Avg speed: {avg_speed:.2f} m/s")

    def run(self):
        """Run the complete simulation"""
        print("\n" + "="*70)
        print("🚦 TRAFI SIMULATION STARTING")
        print("="*70 + "\n")
        
        for step in range(self.total_steps):
            self.run_simulation_step(step)
        
        print("\n" + "="*70)
        print("✅ SIMULATION COMPLETED SUCCESSFULLY")
        print("="*70)
        
        return self.stats
    
    def visualize_results(self, save_figure=False):
        """Create comprehensive visualization dashboard"""
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle('TRAFI - Traffic Flow Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Space-Time Diagram
        ax1 = plt.subplot(3, 3, 1)
        colors = plt.cm.tab20(np.linspace(0, 1, min(30, len(self.finished_vehicles))))
        for idx, vehicle in enumerate(self.finished_vehicles[:30]):
            if vehicle["trajectory"]:
                times = [t["time"] for t in vehicle["trajectory"]]
                positions = [t["position"] + t.get("road_idx", 0) * self.road_length 
                            for t in vehicle["trajectory"]]
                ax1.plot(times, positions, color=colors[idx], alpha=0.7, linewidth=1)
        ax1.set_xlabel("Time (s)", fontsize=10)
        ax1.set_ylabel("Distance (m)", fontsize=10)
        ax1.set_title("Space-Time Diagram", fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Fundamental Diagram
        ax2 = plt.subplot(3, 3, 2)
        if self.stats.time_series_data["density"] and self.stats.time_series_data["flow"]:
            scatter = ax2.scatter(self.stats.time_series_data["density"], 
                       self.stats.time_series_data["flow"], 
                       c=self.stats.time_series_data["speed"], 
                       cmap='viridis', alpha=0.6, s=20)
            ax2.set_xlabel("Density (veh/km)", fontsize=10)
            ax2.set_ylabel("Flow (veh/h)", fontsize=10)
            ax2.set_title("Fundamental Diagram", fontsize=12, fontweight='bold')
            plt.colorbar(scatter, ax=ax2, label='Speed (m/s)')
        
        # 3. Speed Distribution
        ax3 = plt.subplot(3, 3, 3)
        all_speeds = []
        for vehicle in self.finished_vehicles:
            for point in vehicle["trajectory"]:
                all_speeds.append(point["speed"])
        
        if all_speeds:
            ax3.hist(all_speeds, bins=30, alpha=0.7, color='skyblue', 
                    edgecolor='black', density=True)
            ax3.axvline(np.mean(all_speeds), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(all_speeds):.2f} m/s')
            ax3.set_xlabel("Speed (m/s)", fontsize=10)
            ax3.set_ylabel("Density", fontsize=10)
            ax3.set_title("Speed Distribution", fontsize=12, fontweight='bold')
            ax3.legend(fontsize=9)
        
        # 4. Travel Time Distribution
        ax4 = plt.subplot(3, 3, 4)
        if self.stats.travel_times:
            ax4.hist(self.stats.travel_times, bins=20, alpha=0.7, 
                    color='lightgreen', edgecolor='black')
            ax4.axvline(np.mean(self.stats.travel_times), color='red', 
                       linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(self.stats.travel_times):.1f} s')
            ax4.set_xlabel("Travel Time (s)", fontsize=10)
            ax4.set_ylabel("Frequency", fontsize=10)
            ax4.set_title("Travel Time Distribution", fontsize=12, fontweight='bold')
            ax4.legend(fontsize=9)
        
        # 5. Queue Length Over Time
        ax5 = plt.subplot(3, 3, 5)
        if self.stats.time_series_data["queue_length"]:
            ax5.plot(self.stats.time_series_data["queue_length"], 
                    color='darkorange', linewidth=2)
            ax5.fill_between(range(len(self.stats.time_series_data["queue_length"])),
                            self.stats.time_series_data["queue_length"], 
                            alpha=0.3, color='orange')
            ax5.set_xlabel("Time Step", fontsize=10)
            ax5.set_ylabel("Queue Length (veh)", fontsize=10)
            ax5.set_title("Queue Length Over Time", fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3)
        
        # 6. Delay Distribution
        ax6 = plt.subplot(3, 3, 6)
        if self.stats.delays:
            bp = ax6.boxplot(self.stats.delays, patch_artist=True,
                           boxprops=dict(facecolor='lightblue'))
            ax6.set_ylabel("Delay (s)", fontsize=10)
            ax6.set_title("Delay Distribution", fontsize=12, fontweight='bold')
            ax6.grid(True, alpha=0.3, axis='y')
            
            # Add statistics text
            stats_text = (f'Min: {np.min(self.stats.delays):.1f}s\n'
                         f'Q1: {np.percentile(self.stats.delays, 25):.1f}s\n'
                         f'Median: {np.median(self.stats.delays):.1f}s\n'
                         f'Q3: {np.percentile(self.stats.delays, 75):.1f}s\n'
                         f'Max: {np.max(self.stats.delays):.1f}s')
            ax6.text(0.02, 0.98, stats_text, transform=ax6.transAxes,
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 7. Speed Over Time
        ax7 = plt.subplot(3, 3, 7)
        if self.stats.time_series_data["speed"]:
            ax7.plot(self.stats.time_series_data["speed"], 
                    color='green', linewidth=2, alpha=0.8)
            ax7.axhline(np.mean(self.stats.time_series_data["speed"]), 
                       color='red', linestyle='--', 
                       label=f'Avg: {np.mean(self.stats.time_series_data["speed"]):.2f} m/s')
            ax7.set_xlabel("Time Step", fontsize=10)
            ax7.set_ylabel("Average Speed (m/s)", fontsize=10)
            ax7.set_title("System Speed Over Time", fontsize=12, fontweight='bold')
            ax7.legend(fontsize=9)
            ax7.grid(True, alpha=0.3)
        
        # 8. Flow Over Time
        ax8 = plt.subplot(3, 3, 8)
        if self.stats.time_series_data["flow"]:
            ax8.plot(self.stats.time_series_data["flow"], 
                    color='purple', linewidth=2, alpha=0.8)
            if len(self.stats.time_series_data["flow"]) > 0:
                avg_flow = np.mean(self.stats.time_series_data["flow"])
                ax8.axhline(avg_flow, color='red', linestyle='--', 
                           label=f'Avg: {avg_flow:.0f} veh/h')
            ax8.set_xlabel("Time Step", fontsize=10)
            ax8.set_ylabel("Flow (veh/h)", fontsize=10)
            ax8.set_title("Traffic Flow Over Time", fontsize=12, fontweight='bold')
            ax8.legend(fontsize=9)
            ax8.grid(True, alpha=0.3)
        
        # 9. Summary Statistics
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        if self.finished_vehicles:
            summary_text = []
            summary_text.append("SIMULATION SUMMARY")
            summary_text.append("=" * 30)
            summary_text.append(f"Total Vehicles: {len(self.finished_vehicles)}")
            summary_text.append(f"Simulation Time: {self.total_steps * self.dt:.0f}s")
            summary_text.append("")
            summary_text.append("--- PERFORMANCE ---")
            if self.stats.travel_times:
                summary_text.append(f"Avg Travel Time: {np.mean(self.stats.travel_times):.1f}s")
                summary_text.append(f"Avg Speed: {np.mean(self.stats.average_speeds)*3.6:.1f} km/h")
                summary_text.append(f"Avg Delay: {np.mean(self.stats.delays):.1f}s")
            
            if self.stats.waiting_times:
                summary_text.append(f"Avg Wait Time: {np.mean(self.stats.waiting_times):.1f}s")
            
            if self.stats.fuel_consumption:
                summary_text.append("")
                summary_text.append("--- ENVIRONMENT ---")
                summary_text.append(f"Total Fuel: {sum(self.stats.fuel_consumption):.1f} L")
                summary_text.append(f"Total CO2: {sum(self.stats.emissions)/1000:.1f} kg")
            
            ax9.text(0.05, 0.95, "\n".join(summary_text), transform=ax9.transAxes,
                    verticalalignment='top', fontsize=9, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_figure:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"TRAFI_dashboard_{timestamp}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"\n📊 Dashboard saved as: {filename}")
        
        plt.show()
    
    def export_results(self, filename_prefix="TRAFI_results"):
        """Export simulation results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export trajectories
        trajectories = {
            "metadata": {
                "simulation_name": "TRAFI",
                "timestamp": timestamp,
                "time_step": self.dt,
                "total_steps": self.total_steps,
                "road_length": self.road_length,
                "car_following_model": self.car_following_model,
                "enable_signals": self.enable_signals
            },
            "vehicles": []
        }
        
        for vehicle in self.finished_vehicles:
            trajectories["vehicles"].append({
                "id": vehicle["id"],
                "route": vehicle["route"],
                "start_time": vehicle["start_time"],
                "total_travel_time": vehicle["trajectory"][-1]["time"] if vehicle["trajectory"] else 0,
                "total_distance": vehicle["total_distance"],
                "average_speed": vehicle["total_distance"] / (vehicle["trajectory"][-1]["time"] if vehicle["trajectory"] else 1),
                "waiting_time": vehicle["waiting_time"],
                "trajectory": vehicle["trajectory"]
            })
        
        traj_filename = f"{filename_prefix}_trajectories_{timestamp}.json"
        with open(traj_filename, 'w') as f:
            json.dump(trajectories, f, indent=2)
        
        # Export statistics
        stats = {
            "travel_times": self.stats.travel_times,
            "average_speeds": self.stats.average_speeds,
            "delays": self.stats.delays,
            "waiting_times": self.stats.waiting_times,
            "fuel_consumption": self.stats.fuel_consumption,
            "emissions": self.stats.emissions,
            "time_series": self.stats.time_series_data
        }
        
        stats_filename = f"{filename_prefix}_statistics_{timestamp}.json"
        with open(stats_filename, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n💾 Results exported to:")
        print(f"   • {traj_filename}")
        print(f"   • {stats_filename}")

    def get_rl_state(self):
        avg_speed = self.stats.time_series_data["speed"][-1] if self.stats.time_series_data["speed"] else 0
        queue = self.stats.time_series_data["queue_length"][-1] if self.stats.time_series_data["queue_length"] else 0
        density = self.stats.time_series_data["density"][-1] if self.stats.time_series_data["density"] else 0
        return np.array([avg_speed, queue, density], dtype=np.float32)
    # ----------------------------- 
    # RL multi-intersection step
    # -----------------------------
    def rl_step(self, step):
        """
        Multi-intersection RL control
        """
        sim_time = step * self.dt

        for iid, signal in self.traffic_signals.items():

            # -------- STATE --------
            queue = 0
            speed_sum = 0
            count = 0

            for road in INTERSECTIONS[iid]:
                vehicles = self.road_network.get_vehicles_on_road(road)
                for v in vehicles:
                    speed_sum += v["speed"]
                    count += 1
                    if v["speed"] < 0.1:  # stopped vehicle
                        queue += 1

            avg_speed = speed_sum / max(1, count)

            # -------- ACTION --------
            # Random policy for now (replace with PPO/DQN later)
            action = np.random.choice([0, 1])

            # Manual phase switch
            current_phase = signal.get_phase(0)  # we can use road 0 for reference
            if action == 1:
                if current_phase == "green":
                    signal.current_time = (signal.current_time + signal.green_time) % signal.cycle_time
                else:
                    signal.current_time = (signal.current_time + (signal.cycle_time - signal.green_time)) % signal.cycle_time

            # -------- REWARD --------
            reward = 0.6 * avg_speed - 1.2 * queue

            # (Optional: store reward for learning later)


    def get_rl_reward(self):
        avg_speed = self.stats.time_series_data["speed"][-1] if self.stats.time_series_data["speed"] else 0
        queue = self.stats.time_series_data["queue_length"][-1] if self.stats.time_series_data["queue_length"] else 0
        delay = np.mean(self.stats.delays[-5:]) if self.stats.delays else 0
        stopped = sum(1 for v in self.vehicles if v["speed"] < 0.1)

        reward = (
            0.6 * avg_speed
            - 1.2 * queue
            - 0.4 * delay
            - 0.2 * stopped
        )

        return reward

    def rl_step(self, step):
        state = self.get_rl_state()

        # Temporary random policy (replace with PPO/DQN)
        action = np.random.choice([0, 1])

        self.traffic_signal.apply_action(action, step * self.dt)

        reward = self.get_rl_reward()

        return state, action, reward

    def get_intersection_id(self, road_name):
        for iid, roads in INTERSECTIONS.items():
            if road_name in roads:
                return iid
        return None



# -------------------------
# Command Line Interface
# -------------------------
def main():
    """Main function to run TRAFI simulator from command line"""
    parser = argparse.ArgumentParser(
        description="TRAFI - Traffic Flow Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python trafi.py --flow flow_3_4.json
  python trafi.py --flow flow_3_4.json --steps 500 --dt 0.5 --model IDM
  python trafi.py --flow flow_3_4.json --no-signals --export
  python trafi.py --steps 200 --road-length 150 --model GIPPS --visualize
        """
    )
    
    parser.add_argument("--flow", type=str, default=None,
                       help="Path to flow JSON file (optional, uses sample data if not provided)")
    parser.add_argument("--steps", type=int, default=300,
                       help="Number of simulation steps (default: 300)")
    parser.add_argument("--dt", type=float, default=1.0,
                       help="Time step in seconds (default: 1.0)")
    parser.add_argument("--road-length", type=float, default=100.0,
                       help="Length of each road segment in meters (default: 100)")
    parser.add_argument("--model", type=str, default="IDM",
                       choices=["IDM", "GIPPS", "SAFE"],
                       help="Car-following model to use (default: IDM)")
    parser.add_argument("--no-signals", action="store_true",
                       help="Disable traffic signals")
    parser.add_argument("--visualize", action="store_true",
                       help="Show visualization dashboard")
    parser.add_argument("--export", action="store_true",
                       help="Export results to JSON files")
    parser.add_argument("--save-plot", action="store_true",
                       help="Save visualization as PNG file")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚗 TRAFI - Traffic Flow Simulator")
    print("="*70)
    
    try:
        # Initialize simulator
        simulator = TRAFI_Simulator(
            flow_file=args.flow,
            dt=args.dt,
            total_steps=args.steps,
            road_length=args.road_length,
            car_following_model=args.model,
            enable_signals=not args.no_signals
        )
        
        # Run simulation
        statistics = simulator.run()
        
        # Display results
        statistics.print_summary()
        
        # Visualize if requested
        if args.visualize:
            print("\n📈 Generating visualization dashboard...")
            simulator.visualize_results(save_figure=args.save_plot)
        
        # Export results if requested
        if args.export:
            simulator.export_results()
        
        print("\n🎉 TRAFI simulation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Simulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# -------------------------
# Run as standalone script
# -------------------------
if __name__ == "__main__":
    main()