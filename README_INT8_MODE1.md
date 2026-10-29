# Mode 1 INT8 package

Mode 1 exports NAI4 version 4 packed INT8 weights, bias/multiplier/shift metadata, and a training-derived StandardScaler. Both FP32 and INT8 models use the same **5 Hz LPF** (`cfg.bin` flag `0x0008`), with a **0.5 Hz HPF on direct gyroscope classification features only** (`cfg.bin` flag `0x0020` for `gyro`/`accel+gyro`). INT8 adds flag `0x0002`. The firmware rejects packages trained with another preprocessing contract.

The Metrics tab reports separate FP32 and simulated INT8 validation performance, including per-class precision/recall/F1 and confusion matrices; these are not hardware-ground-truth results.

See [README_INPUT_PREPROCESSING.md](README_INPUT_PREPROCESSING.md) for the exact five input kinds and feature ordering.
