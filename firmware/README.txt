NoodleAI (NAI)
==============

Drop-in changed files
---------------------
python_gui/app.py
python_gui/ble_link.py   
python_gui/noodle_model.py
python_gui/motion_features.py  
firmware/src/main.cpp
firmware/lib/NoodleAIModel/NoodleAIModel.h
firmware/lib/NoodleAIModel/NoodleAIModel.cpp

NAI representation codes
------------------------------------
1  accel                  ax ay az
2  gyro                   gx gy gz
3  accel+gyro             ax ay az gx gy gz
4  quaternion             relative [qw qx qy qz]
5  velocity               estimated [vx vy vz] m/s
6  velocity+quaternion    [vx vy vz qw qx qy qz]

Derived-feature pipeline
------------------------
Raw QMI8658 accel+gyro at 50 Hz
  -> initialize attitude from the first accelerometer sample (yaw is arbitrary)
  -> 6-axis Madgwick update, beta = 0.10
  -> relative quaternion q_rel = conj(q0) * q
  -> rotate acceleration to the gesture/world frame and subtract 1 g
  -> trapezoidal integration for estimated velocity
  -> linear endpoint drift correction so v(0)=v(T)=0
  -> temporal resampling to the model's normalized length
  -> StandardScaler
  -> Noodle file-backed MLP

Raw accel/gyro modes keep the previous behavior:
  temporal resampling -> per-gesture channel centering -> StandardScaler -> MLP

Data
----
NAI saves raw variable-length six-axis gestures using raw_data + raw_offsets in addition to the normalized data.

Live BLE
--------
The previously restored continuous six-axis BLE streaming path is preserved.
Model upload still pauses live streaming while FFat owns the BLE link.

Suggested first test
--------------------
1. Flash the NAI firmware files together.
2. Start the NAI GUI.
3. Confirm the Live IMU plot moves immediately after connection.
4. Record a fresh raw dataset.
5. Train the same dataset with Accelerometer, Estimated Velocity, and
   Velocity + Quaternion for a controlled comparison.
