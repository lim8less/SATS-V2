import json
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Ensure stdout uses utf-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

def load_and_preprocess_data():
    """Load and preprocess data with improved efficiency"""
    print("Loading and preprocessing data...")
    start_time = time.time()
    
    # Load data with optimized settings
    data = pd.read_csv('traffic_data_baseline.csv', encoding='utf-8')

    # Add derived features for better prediction
    data['traffic_density'] = data['car_count'] / (data['car_count'].max() + 1e-6)
    data['congestion_level'] = data['waiting_time'] / (data['waiting_time'].max() + 1e-6)
    data['efficiency_score'] = data['car_count'] / (data['waiting_time'] + 1e-6)
    
    # Enhanced feature set
    numeric_cols = ['car_count', 'waiting_time', 'traffic_density', 'congestion_level', 'efficiency_score']
    
    print(f"Data preprocessing completed in {time.time() - start_time:.2f} seconds")
    return data, numeric_cols

def create_optimized_dataset(dataset, look_back=5):
    """Create optimized dataset with vectorized operations"""
    X, Y = [], []
    n_features = dataset.shape[1]
    
    # Vectorized approach for better performance
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), :])
        Y.append(dataset[i + look_back, 1])  # Predict waiting time
    
    return np.array(X), np.array(Y)

def build_optimized_model(input_shape, learning_rate=0.001):
    """Build an optimized LSTM model with better architecture"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape, dropout=0.2),
        BatchNormalization(),
        LSTM(64, return_sequences=False, dropout=0.2),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def train_model_with_callbacks(model, X_train, Y_train, validation_split=0.2):
    """Train model with callbacks for better convergence"""
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
    ]
    
    history = model.fit(
        X_train, Y_train,
        epochs=50,
        batch_size=64,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )
    return history

def evaluate_predictions(y_true, y_pred, junction_id):
    """Evaluate model performance with multiple metrics"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"Junction {junction_id} - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    return {'mse': mse, 'mae': mae, 'r2': r2}

def main():
    """Main function with optimized workflow"""
    print("Starting optimized algorithm...")
    total_start_time = time.time()
    
    # Load and preprocess data
    data, numeric_cols = load_and_preprocess_data()
    junction_ids = data['junction_id'].unique().tolist()
    scalers = {}

    # Optimized normalization with StandardScaler for better performance
    print("Normalizing data...")
    for junction_id in junction_ids:
        junction_data = data[data['junction_id'] == junction_id][numeric_cols]
        scaler = StandardScaler()
        scalers[junction_id] = scaler
        data.loc[data['junction_id'] == junction_id, numeric_cols] = scaler.fit_transform(junction_data)

    # Save scalers
    joblib.dump(scalers, 'scalers.pkl')

    # Prepare data with optimized look_back
    look_back = 5  # Increased for better temporal patterns
    X_list, Y_list = [], []

    print("Creating training datasets...")
    for junction_id in junction_ids:
        junction_data = data[data['junction_id'] == junction_id][numeric_cols].values
        if len(junction_data) > look_back + 10:  # Ensure enough data
            X_junction, Y_junction = create_optimized_dataset(junction_data, look_back)
            X_list.append(X_junction)
            Y_list.append(Y_junction)
        else:
            print(f"Insufficient data for junction {junction_id}")

    if not X_list or not Y_list:
        print("Not enough data to train the model.")
        return
    
    # Combine datasets
    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    
    print(f"Training data shape: {X.shape}, Target shape: {Y.shape}")

    # Build and train optimized model
    print("Building and training model...")
    model = build_optimized_model((look_back, len(numeric_cols)))
    history = train_model_with_callbacks(model, X, Y)

    # Save model
    model.save('traffic_light_model.h5')

    # Generate predictions with improved efficiency
    print("Generating predictions...")
    predictions_dict = {}
    performance_metrics = {}

    for junction_id in junction_ids:
        junction_data = data[data['junction_id'] == junction_id][numeric_cols].values
        if len(junction_data) > look_back:
            X_junction, Y_junction = create_optimized_dataset(junction_data, look_back)
            
            # Batch prediction for efficiency
            predictions = model.predict(X_junction, batch_size=64, verbose=0)
            
            # Inverse transform predictions
            scaler = scalers[junction_id]
            predictions_reshaped = np.zeros((predictions.shape[0], len(numeric_cols)))
            predictions_reshaped[:, 1] = predictions.flatten()  # waiting_time column
            predictions_inverse = scaler.inverse_transform(predictions_reshaped)[:, 1]
            
            predictions_dict[junction_id] = predictions_inverse.tolist()
            
            # Evaluate performance
            true_values = junction_data[look_back:, 1]  # Actual waiting times
            metrics = evaluate_predictions(true_values, predictions_inverse, junction_id)
            performance_metrics[junction_id] = metrics
    
    # Save results
    results = {
        'junction_ids': junction_ids,
        'predictions': predictions_dict,
        'performance_metrics': performance_metrics,
        'model_info': {
            'look_back': look_back,
            'features': numeric_cols,
            'training_time': time.time() - total_start_time
        }
    }
    
    with open('predictions.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Calculate overall efficiency improvement
    avg_r2 = np.mean([metrics['r2'] for metrics in performance_metrics.values()])
    avg_mae = np.mean([metrics['mae'] for metrics in performance_metrics.values()])
    
    print(f"\n=== ALGORITHM EFFICIENCY SUMMARY ===")
    print(f"Total execution time: {time.time() - total_start_time:.2f} seconds")
    print(f"Average R² Score: {avg_r2:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    print(f"Number of junctions processed: {len(junction_ids)}")
    
    # Create improved visualization
    create_efficiency_visualization(data, predictions_dict, performance_metrics, look_back)
    
    return results

def create_efficiency_visualization(data, predictions_dict, performance_metrics, look_back):
    """Create comprehensive efficiency visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # Plot 1: True vs Predicted for all junctions
    ax1 = axes[0, 0]
    for junction_id in predictions_dict.keys():
        junction_data = data[data['junction_id'] == junction_id]
        true_data = junction_data['waiting_time'].values[look_back:]
        predictions = predictions_dict[junction_id][:len(true_data)]
        
        ax1.plot(true_data, alpha=0.7, label=f'True {junction_id}')
        ax1.plot(predictions, '--', alpha=0.7, label=f'Pred {junction_id}')
    
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Waiting Time')
    ax1.set_title('True vs Predicted Waiting Times')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Performance metrics comparison
    ax2 = axes[0, 1]
    junctions = list(performance_metrics.keys())
    r2_scores = [performance_metrics[j]['r2'] for j in junctions]
    mae_scores = [performance_metrics[j]['mae'] for j in junctions]
    
    x = np.arange(len(junctions))
    width = 0.35
    
    ax2.bar(x - width/2, r2_scores, width, label='R² Score', alpha=0.8)
    ax2.bar(x + width/2, mae_scores, width, label='MAE', alpha=0.8)
    ax2.set_xlabel('Junctions')
    ax2.set_ylabel('Score')
    ax2.set_title('Performance Metrics by Junction')
    ax2.set_xticks(x)
    ax2.set_xticklabels(junctions)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Traffic density vs waiting time
    ax3 = axes[1, 0]
    for junction_id in data['junction_id'].unique():
        junction_data = data[data['junction_id'] == junction_id]
        ax3.scatter(junction_data['car_count'], junction_data['waiting_time'], 
                   alpha=0.6, label=junction_id)
    
    ax3.set_xlabel('Car Count')
    ax3.set_ylabel('Waiting Time')
    ax3.set_title('Traffic Density vs Waiting Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Efficiency improvement over time
    ax4 = axes[1, 1]
    all_efficiency = []
    all_predictions = []
    
    for junction_id in predictions_dict.keys():
        junction_data = data[data['junction_id'] == junction_id]
        true_efficiency = junction_data['efficiency_score'].values[look_back:]
        predictions = predictions_dict[junction_id][:len(true_efficiency)]
        
        # Calculate predicted efficiency (inverse relationship with waiting time)
        pred_efficiency = 1 / (np.array(predictions) + 1e-6)
        
        all_efficiency.extend(true_efficiency)
        all_predictions.extend(pred_efficiency)
    
    ax4.plot(all_efficiency, label='True Efficiency', alpha=0.7)
    ax4.plot(all_predictions, '--', label='Predicted Efficiency', alpha=0.7)
    ax4.set_xlabel('Time Steps')
    ax4.set_ylabel('Efficiency Score')
    ax4.set_title('Traffic Efficiency Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('efficiency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
