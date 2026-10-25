# Mode 1 packed INT8 export and BLE deployment

The browser trains its MLP in TensorFlow.js, calibrates full INT8 using **training-only** standardized inputs, and evaluates its quantized predictor on held-out validation data. The exported INT8 NAI4 package contains `cfg.bin` with `FLAG_INT8` and `Q8M1` metadata, `mean.bin`, `scale.bin`, and **two files per Dense layer**: `wNN.bin` and `pNN.bin`.

- `wNN.bin`: signed INT8, output-major `[O][I]` weights.
- `pNN.bin`: for each output neuron, three little-endian INT32 integers in order `(bias, multiplier, shift)`; exactly 12 bytes per neuron.
- No split `bNN.bin`, `mNN.bin`, `sNN.bin` files in INT8 packages.
- The original FP32 model export (`wNN.bin` and `bNN.bin`) remains unchanged.

The browser deploy selector defaults to INT8, but deployment is refused unless the connected firmware announces `NAI_PRECISION:INT8` over the BLE status characteristic. This prevents corrupting a working FP32 deployment. The INT8 firmware is provided in the adjacent `firmware/` folder under PlatformIO environment `esp32-s3-matrix-int8`.

The browser's binary output uses Noodle's 1/256-step sigmoid LUT. In multiclass prediction it uses the Noodle-compatible INT8 softmax approximation. The new release includes direct host integration tests against the **actual Noodle INT8 FCN implementation** using the supplied reference source.

**Hardware status:** FP32 emlearn benchmarking is unaffected; ESP32-S3 INT8 firmware build/flash and over-the-air BLE deployment need verification on the physical board.

## Paper metrics (FP32 and INT8)

The Metrics tab now contains **two separately labelled validation panels** (FP32 final epoch and post-training INT8), with per-class metrics and confusion matrices for both. The *Export paper report (.zip)* action includes `per_class_int8_validation.csv`, `confusion_matrix_int8_validation.csv`, `validation_predictions_int8.csv`, and `fp32_int8_comparison.csv`. The FP32/INT8 class agreement is measured on identical held-out gestures. The INT8 scores use the JavaScript quantized inference simulator, not device-ground-truth evaluation. Existing datasets can be reloaded and retrained, but the original FP32 paper results ZIP is not sufficient to retroactively reconstruct INT8 results.
