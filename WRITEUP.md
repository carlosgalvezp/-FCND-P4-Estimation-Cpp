# Project: Estimation

---


# [Rubric](https://review.udacity.com/#!/rubrics/1807/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
## Writeup / README

### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it! Below I describe how I addressed each rubric point and where in my code each point is handled.

## Implement Estimator

### 1. Determine the standard deviation of the measurement noise of both GPS X data and Accelerometer X data.
To accomplish this, we first run scenario `06_NoisySensors`. Here, the quad is standing still
with 0 acceleration and position (0,0). Therefore we would expect the measurements from GPS and
accelerometer to be close to 0.

After running this test, we obtain 2 CSV files `config/log/Graph1.txt` and `config/log/Graph2.txt`
containing the received data for each timestep.

Finally, we implement a simple Python script that parses this CSV file and computes
the standard deviation of the received measurements (assuming mean = 0.0). The script is shown
below:

```python
import numpy as np

paths = ['../config/log/Graph1.txt', '../config/log/Graph2.txt']

for path in paths:
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    std = np.std(data[:,1])
    print('std for data {}: {}'.format(path, std))
```

And this provides the following result:

```
std for data ../config/log/Graph1.txt: 0.658928052232
std for data ../config/log/Graph2.txt: 0.490257237601
```

Which is very close to the real values of 0.7 and 0.5 for the GPS and accelerometer, respectively,
as can be found in `config/SimulatedSensors.txt`.

With these values, we pass scenario `06_NoisySensors` successfully:

![q1](images/step1.jpg)

```
Simulation #1 (../config/06_SensorNoise.txt)
PASS: ABS(Quad.GPS.X-Quad.Pos.X) was less than MeasuredStdDev_GPSPosXY for 67% of the time
PASS: ABS(Quad.IMU.AX-0.000000) was less than MeasuredStdDev_AccelXY for 68% of the time
```


### 2. Implement a better rate gyro attitude integration scheme in the `UpdateFromIMU()` function.

We implement a non-linear gyro attitude integration based on quaternions, as explained in section
"7.1.2 Non-linear complementary filter" of the document [Estimation for Quadrotors](https://www.overleaf.com/read/vymfngphcccj#/54894644/).

The implementation, in the `UpdateFromIMU` function, is shown below:

```cpp
Quaternion<float> q = Quaternion<float>::FromEuler123_RPY(rollEst, pitchEst, ekfState(6));
Quaternion<float> q_integrated = q.IntegrateBodyRate(gyro, dtIMU);

const float predictedRoll = q_integrated.Roll();
const float predictedPitch = q_integrated.Pitch();
ekfState(6) = q_integrated.Yaw();

// normalize yaw to -pi .. pi
if (ekfState(6) > F_PI) ekfState(6) -= 2.f*F_PI;
if (ekfState(6) < -F_PI) ekfState(6) += 2.f*F_PI;
```

This gives much better results than the linear complementary filter, managing to pass
scenario `07_AttitudeEstimation`:

![q2](images/step2.jpg)

```
Simulation #1 (../config/07_AttitudeEstimation.txt)
PASS: ABS(Quad.Est.E.MaxEuler) was less than 0.100000 for at least 3.000000 seconds
```

### 3. Implement all of the elements of the prediction step for the estimator.

#### State update

First, we implement the `PredictState` function using Equation (49) in section
"7.2 Transition Model" of the document [Estimation for Quadrotors](https://www.overleaf.com/read/vymfngphcccj#/54894644/),
as shown below:

```cpp
// Integrate position
predictedState(0) = curState(0) + curState(3)*dt;
predictedState(1) = curState(1) + curState(4)*dt;
predictedState(2) = curState(2) + curState(5)*dt;

// Integrate velocities using accelerometers as input
const V3F accel_inertial = attitude.Rotate_BtoI(accel);

predictedState(3) = curState(3)                    + accel_inertial.x*dt;
predictedState(4) = curState(4)                    + accel_inertial.y*dt;
predictedState(5) = curState(5) - CONST_GRAVITY*dt + accel_inertial.z*dt;

// Not integrating yaw since that's done in IMU update
predictedState(6) = curState(6);
```

The input `u_t`, which in this case is the acceleration, is rotated from body to world
frame using the matrix `R_bg`. This is easily implemented with the function `Rotate_BtoI`.
Finally, we do not integrate the yaw since that was already done in the IMU update.
The drone can now accurately follow the trajectory:

![q3](images/step3.jpg)

#### Covariance update

Next step is to implement the covariance update, for which we need to compute
the matrix `gPrime` (Equation 51). First, we implement the matrix `Rbg_prime` in the function
`GetRbgPrime`, as follows:

```cpp
// Match notation
const float phi = roll;
const float theta = pitch;
const float psi = yaw;

// Cosines and sines
const float cos_phi = std::cos(phi);
const float sin_phi = std::sin(phi);

const float cos_theta = std::cos(theta);
const float sin_theta = std::sin(theta);

const float cos_psi = std::cos(psi);
const float sin_psi = std::sin(psi);

// Fill the matrix
RbgPrime(0,0) = -cos_theta*sin_psi;
RbgPrime(0,1) = -sin_phi*sin_theta*sin_psi - cos_theta*cos_psi;
RbgPrime(0,2) = -cos_phi*sin_theta*sin_psi + sin_phi*cos_psi;
RbgPrime(1,0) = cos_theta*cos_psi;
RbgPrime(1,1) = sin_phi*sin_theta*cos_psi - cos_phi*sin_psi;
RbgPrime(1,2) = cos_phi*sin_theta*cos_psi + sin_phi*sin_psi;
RbgPrime(2,0) = 0.0F;
RbgPrime(2,1) = 0.0F;
RbgPrime(2,2) = 0.0F;
```

Then we implement the matrix `gPrime`, in the `Predict` function. Since it's initialized to the identity
matrix, we only need to fill the non-zero elements:

```cpp
  // Compute g prime
  gPrime(0, 3) = dt;
  gPrime(1, 4) = dt;
  gPrime(2, 5) = dt;
  gPrime(3, 6) = (RbgPrime(0,0)*accel.x + RbgPrime(0,1)*accel.y + RbgPrime(0,2)*accel.z)*dt;
  gPrime(4, 6) = (RbgPrime(1,0)*accel.x + RbgPrime(1,1)*accel.y + RbgPrime(1,2)*accel.z)*dt;
  gPrime(5, 6) = (RbgPrime(2,0)*accel.x + RbgPrime(2,1)*accel.y + RbgPrime(2,2)*accel.z)*dt;
```

Finally, we compute the new covariance matrix following the EKF covariance update equation
(see Algorithm 2 in the document [Estimation for Quadrotors](https://www.overleaf.com/read/vymfngphcccj#/54894644/)):

```cpp
// Compute new covariance matrix
MatrixXf newCov = gPrime * ekfCov * gPrime.transpose() + Q;
```

Last, the current estimated state and covariance are updated:

```cpp
ekfState = newState;
ekfCov = newCov;
```

To validate this, we run the scenario `09_PredictCovariance`, and observe how it grows with time. We tune
the corresponding process noise to capture the uncertainty accurately.

![q4](images/step4.jpg)


### 4. Implement the magnetometer update.

The magnetometer update is implemented in the function `UpdateFromMag`, following the equations shown
in section 7.3.2 of the document [Estimation for Quadrotors](https://www.overleaf.com/read/vymfngphcccj#/54894644/)):

```cpp
hPrime(6) = 1.0F;
zFromX(0) = ekfState(6);

// Since the Update function will apply: ekfState = ekfState + K * (z - zFromX),
// we need to make sure that z-zFromX is the shortest difference around the circle
const float shortest_diff = normalizeAngle(z(0) - zFromX(0));
zFromX(0) = z(0) - shortest_diff;
```

The only trick here is that the `Update` function performs a subtraction
`z - zFromX` that does not take into account the fact that `z` and `zFromX` can
actually be angles, and the difference should always be the shortest difference
along the unit circle.

To account for this, we compute what this shortest difference should be, using
the `normalizeAngle` function:

```cpp
float normalizeAngle(const float x)
{
    float y = fmodf(x + F_PI, 2.0F*F_PI);

    if (y < 0.0F)
    {
        y += 2.0F*F_PI;
    }

    return y - F_PI;
}
```

Then we simply set `zFromX(0) = z(0) - shortest_diff`, such that we it's
passed to the `Update` function, the result of the substraction will be
`z - zFromX` = `z - (z - shortest_diff)` = `shortest_diff`, which is what
we want.

We verify the implementation running scenario `10_MagUpdate`:



### 5. Implement the GPS update.


## Flight Evaluation

### 1. Meet the performance criteria of each step.

### 2. De-tune your controller to successfully fly the final desired box trajectory with your estimator and realistic sensors.
