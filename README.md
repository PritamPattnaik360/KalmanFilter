# KalmanFilter
This Python script demonstrates the implementation of a Kalman Filter to estimate the position and velocity of an object undergoing linear motion. The Kalman Filter is a recursive algorithm used for state estimation, combining information from a series of measurements over time with predictions generated by a dynamic model.

The plot shows the true position, noisy measurements, and estimated position over time.
![](img/Capture.PNG)
## The Math Behind It
The Kalman Filter is a mathematical method used to estimate the state of a system by combining predictions from a dynamic model with measurements. In the prediction step, we use the system's dynamics to forecast the next state and its uncertainty. This prediction is refined in the update step, where we adjust our estimate based on the difference between the actual and predicted measurements. The Kalman Gain determines how much we trust the measurement versus the prediction, ensuring an optimal balance between the two. By iteratively updating the state estimate and its uncertainty, the Kalman Filter provides an accurate and efficient estimation of the system's state over time.
### Prediction Step:
- Predicted State:
The next state estimate (x̂ₖ) is predicted based on the current state (x̂ₖ₋₁) and the dynamics of the system represented by the state transition matrix (A).
Mathematically:
x̂ₖ = A * x̂ₖ₋₁
- Predicted Covariance:
The covariance matrix (Pₖ) predicts the uncertainty associated with the state estimate for the next time step. It's updated based on the process noise (Q) that accounts for system dynamics.
Mathematically:
Pₖ = A * Pₖ₋₁ * Aᵀ + Q

### Update Step:
- Kalman Gain:
The Kalman Gain (Kₖ) determines how much we trust the measurement versus the prediction. It's calculated based on the predicted covariance (Pₖ), the measurement matrix (H), and the measurement noise (R).
Mathematically:
Kₖ = Pₖ * Hᵀ * (H * Pₖ * Hᵀ + R)⁻¹
- Updated State Estimate:
Using the Kalman Gain, we update the state estimate (x̂ₖ) based on the difference between the actual measurement (zₖ) and the predicted measurement.
Mathematically:
x̂ₖ = x̂ₖ + Kₖ * (zₖ - H * x̂ₖ)
- Updated Covariance:
Finally, after incorporating the measurement, we update the covariance matrix (Pₖ) to reflect the reduced uncertainty.
Mathematically:
Pₖ = (I - Kₖ * H) * Pₖ



