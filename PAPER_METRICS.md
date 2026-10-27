# NoodleAI Mode 1: evaluation reporting guide

## Workflow

1. Record multiple BOOT-delimited gestures for every class, preferably across multiple recording sessions/participants.
2. Select the input representation, FCN hidden layers, and epoch count, then train.
3. In **Results & Curves**, review numeric validation accuracy, macro precision, macro recall, macro F1, weighted F1, per-class values, and the confusion matrix.
4. Click **Export paper report (.zip)** to preserve CSVs, prediction-level evidence, split indices, training curves (PNG), and run metadata. Archive these alongside the dataset for reproducibility.
5. Before final publication, run an independent subject/session-level evaluation separate from the tuning data, and report its metrics independently.

## Suggested table columns

| Representation | Topology | Train n | Validation n | Accuracy (%) | Macro precision | Macro recall | Macro F1 | Weighted F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fill from the exported CSV | Fill from run_metadata.json | n | n | ... | ... | ... | ... | ... |

The numbers displayed in the UI are placeholders until a real training run completes. No performance results are embedded in the application.

## Interpretation

- Accuracy is the fraction of correctly classified gestures.
- Per-class recall answers "of all true gestures in this class, how many were identified?"
- Per-class precision answers "of all predictions of this class, how many were correct?"
- Macro F1 weighs each class equally, regardless of how many gestures it has.
- Weighted F1 weighs each class by the number of true gestures, so larger classes contribute more.
- A confusion matrix reveals which gesture classes are mistaken for each other.

**Warning:** The browser's current split is a random gesture-level holdout, not leave-one-person-out. If multiple gestures come from the same participant or session in both partitions, performance can be optimistic. Do not relabel validation metrics as test accuracy or claim hardware-side equivalence without separate measurements on the ESP32-S3.
