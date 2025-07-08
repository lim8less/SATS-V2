# SATS Traffic Simulation & Optimization

## Project Description
SATS (Synchronized Automated Traffic Signaling) is a research and educational project that demonstrates how modern traffic intersections can be optimized using simulation and machine learning. The system uses the SUMO (Simulation of Urban MObility) platform to simulate real-world traffic, collects data on vehicle flow and waiting times, and then applies a deep learning model (LSTM) to predict and optimize traffic light timings. The project provides a full pipeline: from baseline simulation, through model training, to an optimized simulation and a quantitative comparison of results.

Key features:
- **Realistic traffic simulation** using SUMO and custom network/route files
- **Automated data collection** of vehicle counts, waiting times, and traffic light states
- **LSTM-based machine learning** for predicting optimal green light durations
- **Dynamic generation of SUMO traffic light logic** based on model predictions
- **Automated efficiency comparison** between baseline and optimized scenarios
- **Comprehensive visualizations** and metrics for before/after analysis

---

## Project Structure & File Descriptions
```
SATS-master-main/
├── main.py                       # Orchestrates the full workflow (run this file)
├── algorithm.py                  # ML model training and prediction (uses baseline traffic data)
├── finalSimulation.py            # Generates optimized traffic light logic from model predictions
├── baselineSimulation.py         # Collects baseline traffic data from SUMO
├── baseline_configuration.sumo.cfg # SUMO config for baseline simulation
├── final_configuration.sumo.cfg  # SUMO config for optimized simulation
├── intersection.net.xml          # SUMO network definition (road network)
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

---

## Setup Instructions
### 1. Prerequisites
- **Python 3.11+** (recommended: 3.12 for best compatibility)
- **SUMO** (Simulation of Urban MObility) installed and available in your PATH ([Download here](https://www.eclipse.dev/sumo/))
- **Conda** (recommended) or `venv` for environment management

### 2. Clone the Repository
```bash
git clone https://github.com/lim8less/SATS-V2.git
cd SATS
```

### 3. Install Python Dependencies
```bash
# Create and activate a conda environment (recommended)
conda create -n sats-env python=3.8
conda activate sats-env

# Install required packages
pip install -r requirements.txt
```

### 4. SUMO Setup
- Download and install SUMO from the [official website](https://www.eclipse.dev/sumo/).
- Ensure `sumo-gui` and `sumo` are available in your system PATH (test with `sumo-gui --help`).

---

## How to Run the Project
1. **Start the workflow:**
   ```bash
   python main.py
   ```
2. The script will:
   - Run a baseline SUMO simulation and collect traffic data
   - Train the LSTM model and generate optimized traffic light logic
   - Run the optimized simulation
   - Output efficiency comparison metrics and visualizations
3. **Check the output files:**
   - `efficiency_comparison.json` for a summary of improvements
   - `efficiency_analysis.png` for visualizations
   - Other generated files as described above

---

## Screenshots
<!--
Add screenshots of the SUMO GUI, efficiency_analysis.png, and any other relevant visualizations here.
Example:
![SUMO Simulation](screenshots/sumo_gui.png)
![Efficiency Analysis](screenshots/efficiency_analysis.png)
-->

---

## Concepts & Algorithms
- **SUMO & TraCI**: Real-time traffic simulation and control
- **LSTM (Long Short-Term Memory)**: Sequence modeling for time-series prediction
- **Feature Engineering**: Creating derived features for better model performance
- **Model Evaluation**: MSE, MAE, R², and efficiency metrics
- **Automated Experimentation**: Full pipeline from data collection to evaluation

---

## Troubleshooting
- Ensure SUMO is installed and accessible from your command line.
- Use Python 3.12 for best compatibility with TensorFlow and SUMO TraCI.
- If you encounter missing packages, install them manually using `pip install <package>`.
- If you see JSON serialization errors, ensure all pandas/numpy values are converted to native Python types before writing to JSON.

---

## Credits
Developed for research and educational purposes. Integrates open-source tools: SUMO, Keras/TensorFlow, scikit-learn, pandas, and matplotlib.
