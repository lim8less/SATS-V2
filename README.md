# SATS Traffic Simulation & Optimization

## Overview
This project simulates and optimizes traffic light control at intersections using SUMO (Simulation of Urban MObility) and machine learning. The system collects real-time traffic data, trains an LSTM-based model to predict optimal traffic light phases, and compares the efficiency of the optimized system against a baseline.

## Features
- **SUMO-based traffic simulation** (with TraCI interface)
- **Data collection**: Vehicle counts, waiting times, and traffic light states
- **Machine learning optimization**: LSTM neural network predicts optimal green light durations
- **Automated efficiency comparison**: Quantifies improvements in waiting time and traffic flow
- **Comprehensive visualizations**: Plots and metrics for before/after analysis

## Project Structure & File Descriptions
```
SATS-master-main/
├── algorithm.py                  # ML model training and prediction (uses baseline traffic data)
├── finalSimulation.py            # Generates optimized traffic light logic from model predictions
├── main.py                       # Orchestrates the full workflow (runs all steps)
├── baseline_final_configuration.sumo.cfg# SUMO config for baseline simulation
├── final_configuration.sumo.cfg        # SUMO config for optimized simulation
├── intersection.net.xml         # SUMO network definition (road network)
├── routes.xml                    # SUMO route definitions (vehicle flows)
├── trips.trips.xml               # SUMO trip definitions (optional, for demand generation)
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── temp.py                       # (Optional) Example/test script (e.g., for GUI)
# --- Generated during execution ---
├── traffic_data_baseline.csv     # Collected traffic data from baseline simulation (created on first run)
├── traffic_light_model.h5        # Trained LSTM model
├── predictions.json              # Model predictions and performance metrics
├── traffic_light_logic.xml       # Optimized traffic light logic for SUMO
├── final_simulation_metrics.json # Metrics from optimized simulation
├── efficiency_comparison.json    # Baseline vs. optimized comparison
├── efficiency_analysis.png       # Visual summary of results
└── ...
```

### Main Files Explained
- **main.py**: The entry point. Runs the entire workflow: baseline simulation, ML training, optimized simulation, and efficiency comparison.
- **algorithm.py**: Trains an LSTM model on collected traffic data (from the baseline simulation) and predicts optimal traffic light timings.
- **finalSimulation.py**: Reads model predictions and generates a new SUMO traffic light logic XML file for the optimized simulation.
- **baseline_configuration.sumo.cfg**: SUMO configuration for the baseline simulation (static/default traffic lights).
- **final_configuration.sumo.cfg**: SUMO configuration for the optimized simulation (uses generated traffic_light_logic.xml).
- **intersection.net.xml**: SUMO network file describing the road network and intersections.
- **routes.xml**: Defines vehicle routes and flows for the simulation.
- **trips.trips.xml**: (Optional) Defines individual vehicle trips for demand generation.
- **requirements.txt**: Lists all Python dependencies needed for the project.
- **README.md**: This documentation file.
- **temp.py**: (Optional) Example or test script, e.g., for GUI prototyping.

### Generated Files (Created During Workflow)
- **traffic_data_baseline.csv**: Collected traffic data from the baseline simulation (created on first run).
- **traffic_light_model.h5**: Trained LSTM model for traffic light optimization.
- **predictions.json**: Model predictions and performance metrics for each junction.
- **traffic_light_logic.xml**: Optimized traffic light logic for SUMO, generated from model predictions.
- **final_simulation_metrics.json**: Metrics from the optimized simulation run.
- **efficiency_comparison.json**: Comparison of baseline and optimized simulation results.
- **efficiency_analysis.png**: Visual summary of efficiency improvements and model performance.

---

The above structure and descriptions should help you understand the role of each file, what is required to start, and what will be generated as you run the project.
