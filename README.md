```
# Distance and Direction Estimation using Inertial Sensors

## Overview

This project was developed as part of a Signal Processing course and focuses on estimating distance and direction using inertial sensors, such as accelerometers and gyroscopes. By leveraging data processing techniques, the system achieves accurate navigation through the integration of Euler's equations of motion, Butterworth filters for noise reduction, and a machine learning approach using a Random Forest Regressor to predict velocities and compare them with GPS data.

## Features

- **Distance Calculation**: Computes distance traveled using accelerometer data through double integration of acceleration (Euler approach).
- **Direction Estimation**: Determines orientation by integrating angular velocity from gyroscopes.
- **Noise Reduction**: Employs Butterworth low-pass and high-pass filters to remove noise and unwanted velocity components.
- **Machine Learning**: Uses a Random Forest Regressor to predict velocities based on accelerometer data, validated against GPS measurements.
- **Visualization**: Includes plots for 2D path tracking, noise removal effects, and comparisons between estimated and GPS-based speeds and distances.

## Methodology

1. **Data Acquisition**: Collects acceleration and angular velocity data from inertial sensors.
2. **Distance Calculation**:
   - Velocity: \( v(t) = \int_0^t a(t) \, dt \)
   - Distance: \( s(t) = \int_0^t v(t) \, dt \)
3. **Direction Calculation**: Orientation computed as \( \rho(t) = \int_0^t \omega(t) \, dt \).
4. **Filtering**:
   - Low-pass Butterworth filter to remove high-frequency noise.
   - High-pass Butterworth filter to eliminate unwanted velocity components.
5. **Machine Learning**:
   - Random Forest Regressor to predict velocities.
   - Performance metrics:
     - Best Cross-Validation MSE: 0.03377815299350328
     - Test Mean Squared Error: 0.029
6. **Output**:
   - Distance traveled (Accelerometer): 421.14 m
   - Distance traveled (GPS): 431.36 m
   - Visualizations for path comparison, speed error, and cumulative distance error.

## Experimental Setup

- **Hardware**: Inertial sensors (accelerometer and gyroscope) for data collection.
- **Software**: Python for data processing, filtering, and machine learning.
- **Validation**: GPS data used as ground truth for comparison.

## Results

- **Distance Accuracy**: Achieved close alignment between accelerometer-based (421.14 m) and GPS-based (431.36 m) distance measurements.
- **ML Performance**: Random Forest Regressor provided low MSE, indicating reliable velocity predictions.
- **Visualizations**:
  - 2D path plots from accelerometer data.
  - Before and after noise removal plots.
  - Estimated vs. GPS speed and cumulative distance error over time.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/distance-direction-estimation.git
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the main script to execute all methods:

   ```bash
   python master_analysis.py
   ```

## Requirements

- Python 3.x
- Libraries: `numpy`, `scipy`, `matplotlib`, `scikit-learn`

## Directory Structure

```
distance-direction-estimation/
│
├── data/                           # Raw and processed sensor data
│   ├── bhaskara_to_ground.csv      # Sample dataset
│   ├── round_around_campus.csv     # Sample dataset
│
├── scripts/                        # Python scripts for processing and ML
│   ├── master_analysis.py          # Main script to run all methods
│   ├── basic_method.py             # Basic distance and direction estimation
│   ├── basic_methodv2.py           # Updated basic method
│   ├── imu_ml_pipeline_anim.py     # ML pipeline for velocity prediction
│   ├── ml_model.py                 # ML model implementation
│
├── plots/                          # Generated visualizations
│   ├── fft_accel_mag_xy.png        # FFT plot for accelerometer data (X-Y)
│   ├── fft_accelerometer_ax.png    # FFT plot for accelerometer (X-axis)
│   ├── fft_accelerometer_ay.png    # FFT plot for accelerometer (Y-axis)
│   ├── fft_accelerometer_az.png    # FFT plot for accelerometer (Z-axis)
│   ├── fft_gyroscope_wx.png        # FFT plot for gyroscope (X-axis)
│   ├── fft_gyroscope_wy.png        # FFT plot for gyroscope (Y-axis)
│   ├── fft_gyroscope_wz.png        # FFT plot for gyroscope (Z-axis)
│   ├── filtered_signal.png         # Filtered signal plot
│   ├── report_fig_paths.png        # Path visualization
│   ├── report_fig_speed.png        # Speed comparison plot
│   ├── report_fig_accel.png        # Acceleration plot
│
├── media/                          # Media files (if any)
├── model/                          # Trained ML models
├── analysis_report.pdf             # Project report in PDF format
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies
```

## Usage

1. Place raw sensor data (e.g., `bhaskara_to_ground.csv`, `round_around_campus.csv`) in the `data/` directory.
2. Run the main script to execute all methods:

   ```bash
   python master_analysis.py
   ```

3. View results and plots in the `plots/` directory.

## Contributors

- Unnath Chittimalla (IMT2023621)
- Akshat Mittal (IMT2023606)
- Mada Hemanth (IMT2023581)

## License

This project is licensed under the MIT License. See the LICENSE file for details.
```
