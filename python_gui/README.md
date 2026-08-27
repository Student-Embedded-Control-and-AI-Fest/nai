# NoodleAI Gesture Trainer

Run:

```bash
pip install -r requirements.txt
python app.py
```

Workflow:

1. Connect the ESP32-S3-Matrix.
2. Define labels and lock the dataset setup.
3. Hold BOOT, perform one complete gesture, release BOOT, then save/discard.
4. Choose **Accelerometer**, **Gyroscope**, or **Accel + Gyro** under Sensor input.
5. Train and inspect the Training Curves tab.
6. Deploy the `.nai` model and switch to Inference mode.

The dataset always retains all six channels, so the same recordings can be retrained in any sensor mode.
