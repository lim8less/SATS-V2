import traci
import pandas as pd
import time
import json
from datetime import datetime

# Connect to the running SUMO GUI simulation
traci_port = 9999
traci.init(traci_port)

data = []
efficiency_metrics = []

step = 0
start_time = time.time()  # Start time for the simulation
last_step_with_cars = 0
total_waiting_time = 0
total_vehicles_processed = 0

print("Initial simulation: Collecting traffic data and efficiency metrics...")

while step < 900:  # Arbitrary step limit; can be adjusted
    traci.simulationStep()
    num_vehicles = traci.simulation.getMinExpectedNumber()  # Get the number of remaining vehicles

    if num_vehicles > 0:
        last_step_with_cars = step  # Track the last step where cars were present
    
    # Collect traffic light data
    tls_ids = traci.trafficlight.getIDList()  # Get all traffic light IDs (junctions)
    
    current_step_waiting_time = 0
    current_step_vehicle_count = 0
    
    for tls_id in tls_ids:
        lane_ids = traci.trafficlight.getControlledLanes(tls_id)  # Get lanes controlled by the traffic light
        for lane_id in lane_ids:
            car_count = traci.lane.getLastStepVehicleNumber(lane_id)
            waiting_time = traci.lane.getWaitingTime(lane_id)
            if car_count >= 0:
                data.append([step, tls_id, lane_id, car_count, waiting_time])
                current_step_waiting_time += waiting_time
                current_step_vehicle_count += car_count
    
    # Collect vehicle-level efficiency metrics
    for vehicle_id in traci.vehicle.getIDList():
        vehicle_waiting_time = traci.vehicle.getWaitingTime(vehicle_id)
        current_step_waiting_time += vehicle_waiting_time
        current_step_vehicle_count += 1
    
    total_waiting_time += current_step_waiting_time
    total_vehicles_processed += current_step_vehicle_count
    
    # Record efficiency metrics for this step
    efficiency_metrics.append({
        'step': step,
        'vehicles': current_step_vehicle_count,
        'waiting_time': current_step_waiting_time,
        'efficiency': current_step_vehicle_count / (current_step_waiting_time + 1e-6)
    })
    
    if num_vehicles == 0:
        break  # Stop the simulation when no cars remain
    
    step += 1

# Calculate elapsed time since no more cars remain
elapsed_time = time.time() - start_time
avg_waiting_time = total_waiting_time / (total_vehicles_processed + 1e-6)
overall_efficiency = total_vehicles_processed / (total_waiting_time + 1e-6)

traci.close()

# Log simulation time and efficiency metrics
print(f"Initial Simulation Time (until no cars remain): {elapsed_time:.2f} seconds")
print(f"Total vehicles processed: {total_vehicles_processed}")
print(f"Average waiting time: {avg_waiting_time:.2f} seconds")
print(f"Overall efficiency: {overall_efficiency:.4f}")

# Save data to a CSV file
df = pd.DataFrame(data, columns=['step', 'junction_id', 'lane_id', 'car_count', 'waiting_time'])
df.to_csv('traffic_data_baseline.csv', index=False)

# Save initial simulation metrics for comparison
initial_metrics = {
    'simulation_time': elapsed_time,
    'total_vehicles_processed': total_vehicles_processed,
    'total_waiting_time': total_waiting_time,
    'average_waiting_time': avg_waiting_time,
    'overall_efficiency': overall_efficiency,
    'steps_completed': step,
    'efficiency_data': efficiency_metrics,
    'timestamp': datetime.now().isoformat()
}

with open('baseline_simulation_metrics.json', 'w') as f:
    json.dump(initial_metrics, f, indent=2)

print("Initial simulation metrics saved to: baseline_simulation_metrics.json")
