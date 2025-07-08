import subprocess
import time
import traci
import json
import pandas as pd
import numpy as np
from datetime import datetime

def run_initial_simulation():
    """Run the initial SUMO simulation with fixed speed settings"""
    print("Running initial SUMO simulation on port 9999...")
    print("Simulation settings: Fixed speed for consistent comparison")
    
    # Run SUMO with fixed simulation speed for consistency
    sumo_process = subprocess.Popen([
        "sumo-gui", 
        "-c", "baseline_configuration.sumo.cfg", 
        "--remote-port", "9999",
        "--step-length", "1.0",  # Fixed step length
        "--time-to-teleport", "300"  # Consistent teleport settings
    ])
    
    # Run baselineSimulation.py concurrently
    print("Running baselineSimulation.py...")
    simulation_process = subprocess.Popen(["python", "baselineSimulation.py"])

    # Wait for both processes to complete
    sumo_process.wait()
    simulation_process.wait()

    print("Initial simulation has completed.")

def generate_model_and_predictions():
    """Run the optimized algorithm to process traffic data"""
    print("Running optimized algorithm.py...")
    start_time = time.time()
    subprocess.run(["python", "algorithm.py"])
    algorithm_time = time.time() - start_time
    print(f"Algorithm execution time: {algorithm_time:.2f} seconds")
    return algorithm_time

def generate_traffic_light_logic():
    """Generate optimized traffic light logic"""
    print("Running finalSimulation.py...")
    subprocess.run(["python", "finalSimulation.py"])

def run_final_simulation():
    """Run the final simulation with optimized traffic light logic"""
    print("Starting final SUMO simulation with optimized traffic light logic...")
    
    # Start the final SUMO simulation with TraCI
    traci.start([
        "sumo-gui", 
        "-c", "final_configuration.sumo.cfg",
        "--step-length", "1.0",  # Same step length as initial simulation
        "--time-to-teleport", "300"  # Same teleport settings
    ])

    step = 0
    start_time = time.time()
    last_step_with_cars = 0
    total_waiting_time = 0
    total_vehicles_processed = 0
    efficiency_data = []

    while step < 1000:  # Same limit as initial simulation
        traci.simulationStep()
        num_vehicles = traci.simulation.getMinExpectedNumber()

        if num_vehicles > 0:
            last_step_with_cars = step
        
        # Collect efficiency metrics
        current_waiting_time = 0
        current_vehicle_count = 0
        
        for vehicle_id in traci.vehicle.getIDList():
            current_waiting_time += traci.vehicle.getWaitingTime(vehicle_id)
            current_vehicle_count += 1
        
        total_waiting_time += current_waiting_time
        total_vehicles_processed += current_vehicle_count
        
        efficiency_data.append({
            'step': step,
            'vehicles': current_vehicle_count,
            'waiting_time': current_waiting_time,
            'efficiency': current_vehicle_count / (current_waiting_time + 1e-6)
        })

        if num_vehicles == 0:
            break
        
        step += 1

    # Calculate final metrics
    elapsed_time = time.time() - start_time
    avg_waiting_time = total_waiting_time / (total_vehicles_processed + 1e-6)
    overall_efficiency = total_vehicles_processed / (total_waiting_time + 1e-6)

    traci.close()
    
    # Save final simulation metrics
    final_metrics = {
        'simulation_time': elapsed_time,
        'total_vehicles_processed': total_vehicles_processed,
        'total_waiting_time': total_waiting_time,
        'average_waiting_time': avg_waiting_time,
        'overall_efficiency': overall_efficiency,
        'steps_completed': step,
        'efficiency_data': efficiency_data
    }
    
    with open('final_simulation_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"Final Simulation Time: {elapsed_time:.2f} seconds")
    print(f"Total vehicles processed: {total_vehicles_processed}")
    print(f"Average waiting time: {avg_waiting_time:.2f} seconds")
    print(f"Overall efficiency: {overall_efficiency:.4f}")
    
    return final_metrics

def compare_simulation_efficiency():
    """Compare efficiency between initial and final simulations"""
    print("\n" + "="*60)
    print("EFFICIENCY COMPARISON ANALYSIS")
    print("="*60)
    
    try:
        # Load initial simulation data
        initial_data = pd.read_csv('traffic_data_baseline.csv')
        initial_total_waiting = initial_data['waiting_time'].sum()
        initial_total_vehicles = initial_data['car_count'].sum()
        initial_avg_waiting = initial_data['waiting_time'].mean()
        initial_efficiency = initial_total_vehicles / (initial_total_waiting + 1e-6)
        
        # Load final simulation metrics
        with open('final_simulation_metrics.json', 'r') as f:
            final_metrics = json.load(f)
        
        final_total_waiting = final_metrics['total_waiting_time']
        final_total_vehicles = final_metrics['total_vehicles_processed']
        final_avg_waiting = final_metrics['average_waiting_time']
        final_efficiency = final_metrics['overall_efficiency']
        
        # Calculate improvements
        waiting_time_improvement = ((initial_avg_waiting - final_avg_waiting) / initial_avg_waiting) * 100
        efficiency_improvement = ((final_efficiency - initial_efficiency) / initial_efficiency) * 100
        
        print(f"\nINITIAL SIMULATION (Baseline):")
        print(f"  Total waiting time: {initial_total_waiting:.2f} seconds")
        print(f"  Total vehicles: {initial_total_vehicles}")
        print(f"  Average waiting time: {initial_avg_waiting:.2f} seconds")
        print(f"  Efficiency score: {initial_efficiency:.4f}")
        
        print(f"\nFINAL SIMULATION (Optimized):")
        print(f"  Total waiting time: {final_total_waiting:.2f} seconds")
        print(f"  Total vehicles: {final_total_vehicles}")
        print(f"  Average waiting time: {final_avg_waiting:.2f} seconds")
        print(f"  Efficiency score: {final_efficiency:.4f}")
        
        print(f"\nIMPROVEMENT METRICS:")
        print(f"  Waiting time reduction: {waiting_time_improvement:.2f}%")
        print(f"  Efficiency improvement: {efficiency_improvement:.2f}%")
        
        if waiting_time_improvement > 0:
            print(f"  ✅ ALGORITHM SUCCESS: Reduced waiting time by {waiting_time_improvement:.2f}%")
        else:
            print(f"  ⚠️  ALGORITHM NEEDS IMPROVEMENT: Increased waiting time by {abs(waiting_time_improvement):.2f}%")
        
        if efficiency_improvement > 0:
            print(f"  ✅ ALGORITHM SUCCESS: Improved efficiency by {efficiency_improvement:.2f}%")
        else:
            print(f"  ⚠️  ALGORITHM NEEDS IMPROVEMENT: Decreased efficiency by {abs(efficiency_improvement):.2f}%")
        
        # Save comparison results
        comparison_results = {
            'timestamp': datetime.now().isoformat(),
            'initial_simulation': {
                'total_waiting_time': float(initial_total_waiting),
                'total_vehicles': int(initial_total_vehicles),
                'average_waiting_time': float(initial_avg_waiting),
                'efficiency': float(initial_efficiency)
            },
            'final_simulation': {
                'total_waiting_time': float(final_total_waiting),
                'total_vehicles': int(final_total_vehicles),
                'average_waiting_time': float(final_avg_waiting),
                'efficiency': float(final_efficiency)
            },
            'improvements': {
                'waiting_time_reduction_percent': float(waiting_time_improvement),
                'efficiency_improvement_percent': float(efficiency_improvement)
            }
        }
        
        with open('efficiency_comparison.json', 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        return comparison_results
        
    except Exception as e:
        print(f"Error in efficiency comparison: {e}")
        return None

def main():
    """Main execution function with comprehensive efficiency tracking"""
    print("="*60)
    print("SATS TRAFFIC SIMULATION - EFFICIENCY OPTIMIZATION")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_start_time = time.time()
    
    # Phase 1: Initial simulation (baseline)
    print("\nPHASE 1: Running baseline simulation...")
    run_initial_simulation()
    
    # Phase 2: Algorithm optimization
    print("\nPHASE 2: Running optimization algorithm...")
    algorithm_time = generate_model_and_predictions()
    
    # Phase 3: Generate optimized traffic logic
    print("\nPHASE 3: Generating optimized traffic light logic...")
    generate_traffic_light_logic()
    
    # Phase 4: Final simulation with optimizations
    print("\nPHASE 4: Running optimized simulation...")
    final_metrics = run_final_simulation()
    
    # Phase 5: Efficiency comparison
    print("\nPHASE 5: Analyzing efficiency improvements...")
    comparison_results = compare_simulation_efficiency()
    
    # Final summary
    total_execution_time = time.time() - total_start_time
    
    print(f"\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    print(f"Total execution time: {total_execution_time:.2f} seconds")
    print(f"Algorithm processing time: {algorithm_time:.2f} seconds")
    print(f"Simulation comparison completed successfully")
    print(f"Results saved to: efficiency_comparison.json")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

if __name__ == "__main__":
    main()
