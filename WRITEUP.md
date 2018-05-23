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
void QuadEstimatorEKF::UpdateFromIMU(V3F accel, V3F gyro)
{
  // Improve a complementary filter-type attitude filter
  // 
  // Currently a small-angle approximation integration method is implemented
  // The integrated (predicted) value is then updated in a complementary filter style with attitude information from accelerometers
  // 
  // Implement a better integration method that uses the current attitude estimate (rollEst, pitchEst and ekfState(6))
  // to integrate the body rates into new Euler angles.
  //
  // HINTS:
  //  - there are several ways to go about this, including:
  //    1) create a rotation matrix based on your current Euler angles, integrate that, convert back to Euler angles
  //    OR 
  //    2) use the Quaternion<float> class, which has a handy FromEuler123_RPY function for creating a quaternion from Euler Roll/PitchYaw
  //       (Quaternion<float> also has a IntegrateBodyRate function, though this uses quaternions, not Euler angles)

  ////////////////////////////// BEGIN STUDENT CODE ///////////////////////////
  // SMALL ANGLE GYRO INTEGRATION:
  // (replace the code below)
  // make sure you comment it out when you add your own code -- otherwise e.g. you might integrate yaw twice

  // float predictedPitch = pitchEst + dtIMU * gyro.y;
  // float predictedRoll = rollEst + dtIMU * gyro.x;
  // ekfState(6) = ekfState(6) + dtIMU * gyro.z;	// yaw
  Quaternion<float> q = Quaternion<float>::FromEuler123_RPY(rollEst, pitchEst, ekfState(6));
  Quaternion<float> q_integrated = q.IntegrateBodyRate(gyro, dtIMU);

  const float predictedRoll = q_integrated.Roll();
  const float predictedPitch = q_integrated.Pitch();
  ekfState(6) = q_integrated.Yaw();

  // normalize yaw to -pi .. pi
  if (ekfState(6) > F_PI) ekfState(6) -= 2.f*F_PI;
  if (ekfState(6) < -F_PI) ekfState(6) += 2.f*F_PI;

  /////////////////////////////// END STUDENT CODE ////////////////////////////

  // CALCULATE UPDATE
  accelRoll = atan2f(accel.y, accel.z);
  accelPitch = atan2f(-accel.x, 9.81f);

  // FUSE INTEGRATION AND UPDATE
  rollEst = attitudeTau / (attitudeTau + dtIMU) * (predictedRoll)+dtIMU / (attitudeTau + dtIMU) * accelRoll;
  pitchEst = attitudeTau / (attitudeTau + dtIMU) * (predictedPitch)+dtIMU / (attitudeTau + dtIMU) * accelPitch;

  lastGyro = gyro;
}
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
VectorXf QuadEstimatorEKF::PredictState(VectorXf curState, float dt, V3F accel, V3F gyro)
{
  assert(curState.size() == QUAD_EKF_NUM_STATES);
  VectorXf predictedState = curState;
  // Predict the current state forward by time dt using current accelerations and body rates as input
  // INPUTS: 
  //   curState: starting state
  //   dt: time step to predict forward by [s]
  //   accel: acceleration of the vehicle, in body frame, *not including gravity* [m/s2]
  //   gyro: body rates of the vehicle, in body frame [rad/s]
  //   
  // OUTPUT:
  //   return the predicted state as a vector

  // HINTS 
  // - dt is the time duration for which you should predict. It will be very short (on the order of 1ms)
  //   so simplistic integration methods are fine here
  // - we've created an Attitude Quaternion for you from the current state. Use 
  //   attitude.Rotate_BtoI(<V3F>) to rotate a vector from body frame to inertial frame
  // - the yaw integral is already done in the IMU update. Be sure not to integrate it again here

  Quaternion<float> attitude = Quaternion<float>::FromEuler123_RPY(rollEst, pitchEst, curState(6));

  ////////////////////////////// BEGIN STUDENT CODE ///////////////////////////
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

  /////////////////////////////// END STUDENT CODE ////////////////////////////

  return predictedState;
}
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
MatrixXf QuadEstimatorEKF::GetRbgPrime(float roll, float pitch, float yaw)
{
  // first, figure out the Rbg_prime
  MatrixXf RbgPrime(3, 3);
  RbgPrime.setZero();

  // Return the partial derivative of the Rbg rotation matrix with respect to yaw. We call this RbgPrime.
  // INPUTS: 
  //   roll, pitch, yaw: Euler angles at which to calculate RbgPrime
  //   
  // OUTPUT:
  //   return the 3x3 matrix representing the partial derivative at the given point

  // HINTS
  // - this is just a matter of putting the right sin() and cos() functions in the right place.
  //   make sure you write clear code and triple-check your math
  // - You can also do some numerical partial derivatives in a unit test scheme to check 
  //   that your calculations are reasonable

  ////////////////////////////// BEGIN STUDENT CODE ///////////////////////////
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

  /////////////////////////////// END STUDENT CODE ////////////////////////////

  return RbgPrime;
}
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

The complete `Predict` function is presented here:

```cpp
void QuadEstimatorEKF::Predict(float dt, V3F accel, V3F gyro)
{
  // predict the state forward
  VectorXf newState = PredictState(ekfState, dt, accel, gyro);

  // Predict the current covariance forward by dt using the current accelerations and body rates as input.
  // INPUTS: 
  //   dt: time step to predict forward by [s]
  //   accel: acceleration of the vehicle, in body frame, *not including gravity* [m/s2]
  //   gyro: body rates of the vehicle, in body frame [rad/s]
  //   state (member variable): current state (state at the beginning of this prediction)
  //   
  // OUTPUT:
  //   update the member variable cov to the predicted covariance

  // HINTS
  // - update the covariance matrix cov according to the EKF equation.
  // 
  // - you may find the current estimated attitude in variables rollEst, pitchEst, state(6).
  //
  // - use the class MatrixXf for matrices. To create a 3x5 matrix A, use MatrixXf A(3,5).
  //
  // - the transition model covariance, Q, is loaded up from a parameter file in member variable Q
  // 
  // - This is unfortunately a messy step. Try to split this up into clear, manageable steps:
  //   1) Calculate the necessary helper matrices, building up the transition jacobian
  //   2) Once all the matrices are there, write the equation to update cov.
  //
  // - if you want to transpose a matrix in-place, use A.transposeInPlace(), not A = A.transpose()
  // 

  // we'll want the partial derivative of the Rbg matrix
  MatrixXf RbgPrime = GetRbgPrime(rollEst, pitchEst, ekfState(6));

  // we've created an empty Jacobian for you, currently simply set to identity
  MatrixXf gPrime(QUAD_EKF_NUM_STATES, QUAD_EKF_NUM_STATES);
  gPrime.setIdentity();

  ////////////////////////////// BEGIN STUDENT CODE ///////////////////////////
  // Compute g prime
  gPrime(0, 3) = dt;
  gPrime(1, 4) = dt;
  gPrime(2, 5) = dt;
  gPrime(3, 6) = (RbgPrime(0,0)*accel.x + RbgPrime(0,1)*accel.y + RbgPrime(0,2)*accel.z)*dt;
  gPrime(4, 6) = (RbgPrime(1,0)*accel.x + RbgPrime(1,1)*accel.y + RbgPrime(1,2)*accel.z)*dt;
  gPrime(5, 6) = (RbgPrime(2,0)*accel.x + RbgPrime(2,1)*accel.y + RbgPrime(2,2)*accel.z)*dt;

  // Compute new covariance matrix
  MatrixXf newCov = gPrime * ekfCov * gPrime.transpose() + Q;
  /////////////////////////////// END STUDENT CODE ////////////////////////////

  ekfState = newState;
  ekfCov = newCov;
}
```

To validate this, we run the scenario `09_PredictCovariance`, and observe how it grows with time. We tune
the corresponding process noise to capture the uncertainty accurately.

![q4](images/step4.jpg)


### 4. Implement the magnetometer update.

The magnetometer update is implemented in the function `UpdateFromMag`, following the equations shown
in section 7.3.2 of the document [Estimation for Quadrotors](https://www.overleaf.com/read/vymfngphcccj#/54894644/)):

```cpp
void QuadEstimatorEKF::UpdateFromMag(float magYaw)
{
  VectorXf z(1);
  VectorXf zFromX(1);
  z(0) = magYaw;

  MatrixXf hPrime(1, QUAD_EKF_NUM_STATES);
  hPrime.setZero();

  // MAGNETOMETER UPDATE
  // Hints: 
  //  - Your current estimated yaw can be found in the state vector: ekfState(6)
  //  - Make sure to normalize the difference between your measured and estimated yaw
  //    (you don't want to update your yaw the long way around the circle)
  //  - The magnetomer measurement covariance is available in member variable R_Mag
  ////////////////////////////// BEGIN STUDENT CODE ///////////////////////////
  // Compute h_primer and h(x)
  hPrime(6) = 1.0F;
  zFromX(0) = ekfState(6);

  // Since the Update function will apply: ekfState = ekfState + K * (z - zFromX),
  // we need to make sure that z-zFromX is the shortest difference around the circle
  const float shortest_diff = normalizeAngle(z(0) - zFromX(0));
  zFromX(0) = z(0) - shortest_diff;
  /////////////////////////////// END STUDENT CODE ////////////////////////////

  Update(z, hPrime, R_Mag, zFromX);
}
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

![q5](images/step5.jpg)


```
Simulation #1 (../config/10_MagUpdate.txt)
PASS: ABS(Quad.Est.E.Yaw) was less than 0.120000 for at least 10.000000 seconds
PASS: ABS(Quad.Est.E.Yaw-0.000000) was less than Quad.Est.S.Yaw for 76% of the time
```

### 5. Implement the GPS update.

Last, we implement the GPS update in the function `UpdateFromGPS`, following
section 7.3.1 of the document [Estimation for Quadrotors](https://www.overleaf.com/read/vymfngphcccj#/54894644/)):

```cpp
void QuadEstimatorEKF::UpdateFromGPS(V3F pos, V3F vel)
{
  VectorXf z(6);
  VectorXf zFromX(6);
  z(0) = pos.x;
  z(1) = pos.y;
  z(2) = pos.z;
  z(3) = vel.x;
  z(4) = vel.y;
  z(5) = vel.z;

  MatrixXf hPrime(6, QUAD_EKF_NUM_STATES);
  hPrime.setZero();

  // GPS UPDATE
  // Hints: 
  //  - The GPS measurement covariance is available in member variable R_GPS
  //  - this is a very simple update
  ////////////////////////////// BEGIN STUDENT CODE ///////////////////////////
  for (std::size_t i = 0U; i < 6U; ++i)
  {
    zFromX(i) = ekfState(i);
    hPrime(i, i) = 1.0F;
  }
  /////////////////////////////// END STUDENT CODE ////////////////////////////

  Update(z, hPrime, R_GPS, zFromX);
}
```

This update is very simple, starting from zero-initialized matrices `zFromX` and `hPrime`.
We simply set the measured value to be equal to the estimated value. In addition, we fill with
ones the diagonal of the `hPrime` matrix (first 6 elements).

## Flight Evaluation

### 1. Meet the performance criteria of each step.
All scenarios meet the performance criteria, as shown in the previous sections.


### 2. De-tune your controller to successfully fly the final desired box trajectory with your estimator and realistic sensors.
The controller from P3 has been incorporated and de-tuned, by reducing the position and velocity
gains by 30%. This is enough to make the drone fly with an error of <1m for the entire box flight,
as show in the video and command line output below:

#### Simulation Video

```
Simulation #1 (../config/11_GPSUpdate.txt)
PASS: ABS(Quad.Est.E.Pos) was less than 1.000000 for at least 20.000000 seconds
```

[![Submission Video](http://img.youtube.com/vi/_vUn6wQoVr8/0.jpg)](https://youtu.be/_vUn6wQoVr8)
