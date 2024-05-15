import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, dt, process_var, measurement_var):
        self.dt = dt
        self.process_var = process_var
        self.measurement_var = measurement_var
        self.x = np.zeros((2, 1))  # Initial state (position and velocity)
        self.P = np.eye(2)         # Initial state covariance matrix
        self.Q = np.eye(2) * process_var  # Process noise covariance matrix
        self.R = np.eye(1) * measurement_var  # Measurement noise covariance matrix
        self.A = np.array([[1, dt], [0, 1]])  # State transition matrix
        self.H = np.array([[1, 0]])           # Measurement matrix

    def predict(self):
        # Predict the next state
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        # Update the state based on measurement z
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

# Simulate linear motion
dt = 0.1            # Time step
num_steps = 100     # Number of time steps
process_var = 0.1   # Process noise variance
measurement_var = 1 # Measurement noise variance
true_pos = np.zeros(num_steps)
measurements = np.zeros(num_steps)

# Initial state (position and velocity)
true_pos[0] = 0
true_vel = 2

# Create Kalman filter object
kf = KalmanFilter(dt, process_var, measurement_var)

# Simulate motion and generate measurements
for t in range(1, num_steps):
    true_pos[t] = true_pos[t-1] + true_vel * dt
    measurements[t] = true_pos[t] + np.random.randn() * np.sqrt(measurement_var)

# Run Kalman filter to estimate position and velocity
estimated_pos = np.zeros(num_steps)
for t in range(num_steps):
    kf.predict()
    kf.update(measurements[t])
    estimated_pos[t] = kf.x[0]

# Plot true position, measurements, and estimated position through the Kalman Filter
plt.figure(figsize=(10, 6))
plt.plot(np.arange(num_steps) * dt, true_pos, label='True Position')
plt.scatter(np.arange(num_steps) * dt, measurements, color='red', label='Measurements')
plt.plot(np.arange(num_steps) * dt, estimated_pos, '--', color='green', label='Estimated Position')
plt.xlabel('Time (s)')
plt.ylabel('Position')
plt.title('Kalman Filter for Linear Motion')
plt.legend()
plt.grid(True)
plt.show()
