from __future__ import annotations

import queue
import threading
import warnings
from collections import deque
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from ble_link import BLELink
from noodle_model import (
    NoodleAIFilePackage, NoodleAIReference, export_sklearn_mlp,
    normalize_representation, representation_channel_count,
)
from motion_features import build_representation


FEATURE_CHANNELS = 6  # dataset/live stream always stores all six channels
FEATURE_NAMES = ("ax", "ay", "az", "gx", "gy", "gz")
REPRESENTATION_LABELS = {
    "Accelerometer": "accel",
    "Gyroscope": "gyro",
    "Accel + Gyro": "accel+gyro",
    "Relative Quaternion": "quaternion",
    "Estimated Velocity": "velocity",
    "Velocity + Quaternion": "velocity+quaternion",
}
REPRESENTATION_CHANNEL_NAMES = {
    "accel": "ax,ay,az",
    "gyro": "gx,gy,gz",
    "accel+gyro": "ax,ay,az,gx,gy,gz",
    "quaternion": "qw,qx,qy,qz",
    "velocity": "vx,vy,vz",
    "velocity+quaternion": "vx,vy,vz,qw,qx,qy,qz",
}
RAW_REPRESENTATIONS = {"accel", "gyro", "accel+gyro"}
GESTURE_PREPROCESS_VERSION = "nai4_raw6_plus_motion_repr_v1"
LIVE_SECONDS = 5.0
PLOT_REFRESH_MS = 100
DEFAULT_NORMALIZED_LENGTH = 100
DEFAULT_TRAIN_EPOCHS = 300
TRAIN_PROGRESS_EVERY = 10


def preprocess_gesture(raw: np.ndarray, target_n: int) -> np.ndarray:
    """Resample a complete 6-axis IMU gesture and center every channel.

    Input shape:  [raw_samples, 6]
    Output shape: [target_n * 6], time-major

    [ax0 ay0 az0 gx0 gy0 gz0, ax1 ay1 ...]
    """
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != FEATURE_CHANNELS or len(raw) < 2:
        raise ValueError("Gesture needs at least two 6-axis IMU samples")
    if target_n < 2:
        raise ValueError("Normalized gesture length must be at least 2")

    pos = np.linspace(0.0, float(len(raw) - 1), target_n, dtype=np.float32)
    i0 = np.floor(pos).astype(np.int32)
    i1 = np.minimum(i0 + 1, len(raw) - 1)
    alpha = (pos - i0.astype(np.float32))[:, None]
    out = raw[i0] + alpha * (raw[i1] - raw[i0])

    # Remove only the per-gesture DC component of each channel.  Do not divide
    # by per-gesture standard deviation; motion amplitude remains informative.
    means = np.mean(out, axis=0, dtype=np.float32, keepdims=True)
    out = out - means
    return np.asarray(out, dtype=np.float32).reshape(-1)


class NoodleTrainerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("NoodleAI Gesture Trainer — ESP32-S3-Matrix")
        self.geometry("980x720")
        self.minsize(820, 620)

        self.ble = BLELink()

        self.labels: list[str] = []
        # samples keeps the legacy normalized+centered six-axis vector.
        # raw_samples is the NAI4 source of truth for derived representations.
        self.samples: list[np.ndarray] = []
        self.raw_samples: list[np.ndarray] = []
        self.targets: list[int] = []
        self.raw_lengths: list[int] = []
        self.durations_ms: list[int] = []
        self.setup_locked = False

        self.recording: list[list[float]] | None = None
        self.pending_window: np.ndarray | None = None
        self.pending_raw: np.ndarray | None = None
        self.pending_label_index: int | None = None
        self.pending_raw_length: int | None = None
        self.pending_duration_ms: int | None = None
        self.pending_gesture_end: tuple[int, int] | None = None

        self.device_mode = "?"
        self.model_package: NoodleAIFilePackage | None = None
        self.model: MLPClassifier | None = None
        self.scaler: StandardScaler | None = None

        # Training history for TensorFlow-style diagnostics.
        self.train_epochs: list[int] = []
        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []
        self.train_acc_history: list[float] = []
        self.val_acc_history: list[float] = []

        # Live plot history.  Timestamps are device milliseconds converted to s.
        max_points = 600
        self.live_t = deque(maxlen=max_points)
        self.live_channels = [deque(maxlen=max_points) for _ in range(FEATURE_CHANNELS)]
        self._plot_dirty = False

        self._build_ui()
        self.after(30, self._poll_ble_events)
        self.after(PLOT_REFRESH_MS, self._refresh_plot)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI --
    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        # Device strip: always visible, independent of selected tab.
        device = ttk.LabelFrame(root, text="Device", padding=9)
        device.pack(fill="x", pady=(0, 9))

        self.connect_btn = ttk.Button(device, text="Connect", command=self._toggle_connection)
        self.connect_btn.grid(row=0, column=0, rowspan=2, padx=(0, 10), sticky="ns")

        self.device_status = tk.StringVar(value="Disconnected")
        ttk.Label(device, textvariable=self.device_status).grid(row=0, column=1, columnspan=2, sticky="w")

        self.accel_status = tk.StringVar(value="ax: — g    ay: — g    az: — g")
        self.gyro_status = tk.StringVar(value="gx: — °/s    gy: — °/s    gz: — °/s")
        ttk.Label(device, textvariable=self.accel_status).grid(row=1, column=1, sticky="w", padx=(0, 24))
        ttk.Label(device, textvariable=self.gyro_status).grid(row=1, column=2, sticky="w")
        device.columnconfigure(2, weight=1)

        # The holy tabs.
        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill="both", expand=True)

        self.live_tab = ttk.Frame(self.tabs, padding=9)
        self.dataset_tab = ttk.Frame(self.tabs, padding=9)
        self.curves_tab = ttk.Frame(self.tabs, padding=9)
        self.deploy_tab = ttk.Frame(self.tabs, padding=9)
        self.tabs.add(self.live_tab, text="Live IMU")
        self.tabs.add(self.dataset_tab, text="Dataset & Train")
        self.tabs.add(self.curves_tab, text="Training Curves")
        self.tabs.add(self.deploy_tab, text="Deploy & Infer")

        self._build_live_tab()
        self._build_dataset_tab()
        self._build_training_curves_tab()
        self._build_deploy_tab()

    def _build_live_tab(self) -> None:
        toolbar = ttk.Frame(self.live_tab)
        toolbar.pack(fill="x", pady=(0, 6))
        ttk.Label(toolbar, text="Live QMI8658 — last 5 seconds").pack(side="left")
        self.clear_plot_btn = ttk.Button(toolbar, text="Clear plot", command=self._clear_plot)
        self.clear_plot_btn.pack(side="right")

        self.figure = Figure(figsize=(8.5, 5.4), dpi=100)
        self.ax_accel = self.figure.add_subplot(211)
        self.ax_gyro = self.figure.add_subplot(212, sharex=self.ax_accel)

        self.accel_lines = [
            self.ax_accel.plot([], [], label=name)[0] for name in FEATURE_NAMES[:3]
        ]
        self.gyro_lines = [
            self.ax_gyro.plot([], [], label=name)[0] for name in FEATURE_NAMES[3:]
        ]

        self.ax_accel.set_ylabel("Acceleration [g]")
        self.ax_gyro.set_ylabel("Angular rate [deg/s]")
        self.ax_gyro.set_xlabel("Time [s]")
        self.ax_accel.grid(True, alpha=0.25)
        self.ax_gyro.grid(True, alpha=0.25)
        self.ax_accel.legend(loc="upper right", ncol=3)
        self.ax_gyro.legend(loc="upper right", ncol=3)
        self.figure.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.live_tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw_idle()

    def _build_dataset_tab(self) -> None:
        # Setup ---------------------------------------------------------------
        setup = ttk.LabelFrame(self.dataset_tab, text="1. Dataset setup", padding=10)
        setup.pack(fill="x", pady=(0, 9))

        ttk.Label(setup, text="Normalized gesture length").grid(row=0, column=0, sticky="w")
        self.window_var = tk.IntVar(value=DEFAULT_NORMALIZED_LENGTH)
        self.window_spin = ttk.Spinbox(setup, from_=5, to=500, textvariable=self.window_var, width=8)
        self.window_spin.grid(row=0, column=1, sticky="w", padx=(8, 18))
        self.input_dim_var = tk.StringVar()
        ttk.Label(setup, textvariable=self.input_dim_var).grid(row=0, column=2, sticky="w")
        self.window_var.trace_add("write", lambda *_: self._update_input_dim())
        self._update_input_dim()

        ttk.Label(setup, text="Define all labels first").grid(row=1, column=0, sticky="nw", pady=(9, 0))
        label_box = ttk.Frame(setup)
        label_box.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(9, 0))
        self.label_entry = ttk.Entry(label_box)
        self.label_entry.grid(row=0, column=0, sticky="ew")
        self.label_entry.bind("<Return>", lambda _e: self._add_label())
        self.add_label_btn = ttk.Button(label_box, text="Add label", command=self._add_label)
        self.add_label_btn.grid(row=0, column=1, padx=(8, 0))
        self.remove_label_btn = ttk.Button(label_box, text="Remove", command=self._remove_label)
        self.remove_label_btn.grid(row=0, column=2, padx=(8, 0))
        self.label_list = tk.Listbox(label_box, height=4, exportselection=False)
        self.label_list.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(7, 0))
        label_box.columnconfigure(0, weight=1)

        btnrow = ttk.Frame(setup)
        btnrow.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(9, 0))
        self.lock_btn = ttk.Button(btnrow, text="Lock setup", command=self._lock_setup)
        self.lock_btn.pack(side="left")
        self.reset_btn = ttk.Button(btnrow, text="Reset dataset", command=self._reset_dataset)
        self.reset_btn.pack(side="left", padx=(8, 0))
        self.load_dataset_btn = ttk.Button(btnrow, text="Load dataset", command=self._load_dataset)
        self.load_dataset_btn.pack(side="right")
        self.save_dataset_btn = ttk.Button(btnrow, text="Save dataset", command=self._save_dataset)
        self.save_dataset_btn.pack(side="right", padx=(0, 8))

        # Recording -----------------------------------------------------------
        record = ttk.LabelFrame(self.dataset_tab, text="2. Record examples", padding=10)
        record.pack(fill="x", pady=(0, 9))
        ttk.Label(record, text="Label").grid(row=0, column=0, sticky="w")
        self.record_label_var = tk.StringVar()
        self.record_label_combo = ttk.Combobox(
            record, textvariable=self.record_label_var, state="readonly", width=20
        )
        self.record_label_combo.grid(row=0, column=1, sticky="w", padx=(8, 14))
        self.record_btn = ttk.Button(
            record, text="Use BOOT to record", command=self._start_record, state="disabled"
        )
        self.record_btn.grid(row=0, column=2, padx=(0, 8))
        self.save_sample_btn = ttk.Button(
            record, text="Save sample", command=self._save_sample, state="disabled"
        )
        self.save_sample_btn.grid(row=0, column=3, padx=(0, 8))
        self.discard_btn = ttk.Button(record, text="Discard", command=self._discard_pending, state="disabled")
        self.discard_btn.grid(row=0, column=4)

        self.record_progress = tk.StringVar(
            value="Select a label, then hold BOOT to record; release BOOT to finish."
        )
        ttk.Label(record, textvariable=self.record_progress).grid(
            row=1, column=0, columnspan=5, sticky="w", pady=(7, 0)
        )
        self.dataset_counts = tk.StringVar(value="Dataset: define labels, then lock setup")
        ttk.Label(record, textvariable=self.dataset_counts).grid(
            row=2, column=0, columnspan=5, sticky="w", pady=(4, 0)
        )

        # Train ---------------------------------------------------------------
        train = ttk.LabelFrame(self.dataset_tab, text="3. Train N-layer MLP with scikit-learn", padding=10)
        train.pack(fill="x")

        ttk.Label(train, text="Input representation").grid(row=0, column=0, sticky="w")
        self.representation_var = tk.StringVar(value="Accelerometer")
        self.representation_combo = ttk.Combobox(
            train,
            textvariable=self.representation_var,
            values=list(REPRESENTATION_LABELS.keys()),
            state="readonly",
            width=23,
        )
        self.representation_combo.grid(row=0, column=1, sticky="w", padx=(8, 16))
        self.representation_combo.bind("<<ComboboxSelected>>", lambda _e: self._representation_changed())
        ttk.Label(train, text="Same raw 6-axis dataset; choose raw or derived motion features").grid(
            row=0, column=2, sticky="w"
        )

        ttk.Label(train, text="Hidden layers").grid(row=1, column=0, sticky="w", pady=(7, 0))
        self.hidden_var = tk.StringVar(value="32,16")
        self.hidden_entry = ttk.Entry(train, textvariable=self.hidden_var, width=24)
        self.hidden_entry.grid(row=1, column=1, sticky="w", padx=(8, 16), pady=(7, 0))
        self.hidden_entry.bind("<KeyRelease>", lambda _e: self._update_topology())
        ttk.Label(train, text="Example: 32,16").grid(row=1, column=2, sticky="w", pady=(7, 0))

        ttk.Label(train, text="Epochs").grid(row=2, column=0, sticky="w", pady=(7, 0))
        self.epochs_var = tk.IntVar(value=DEFAULT_TRAIN_EPOCHS)
        self.epochs_spin = ttk.Spinbox(train, from_=10, to=5000, textvariable=self.epochs_var, width=8)
        self.epochs_spin.grid(row=2, column=1, sticky="w", padx=(8, 16), pady=(7, 0))
        ttk.Label(train, text="Curves are measured every epoch").grid(row=2, column=2, sticky="w", pady=(7, 0))

        self.topology_var = tk.StringVar(value="Topology: —")
        ttk.Label(train, textvariable=self.topology_var).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(7, 0)
        )
        self.train_btn = ttk.Button(train, text="Train", command=self._train, state="disabled")
        self.train_btn.grid(row=4, column=0, pady=(9, 0), sticky="w")
        self.train_result = tk.StringVar(value="")
        ttk.Label(train, textvariable=self.train_result).grid(
            row=4, column=1, columnspan=2, padx=(10, 0), pady=(9, 0), sticky="w"
        )

    def _build_training_curves_tab(self) -> None:
        header = ttk.Frame(self.curves_tab)
        header.pack(fill="x", pady=(0, 6))
        self.curves_summary_var = tk.StringVar(
            value="Train a model to see loss and accuracy history."
        )
        ttk.Label(header, textvariable=self.curves_summary_var).pack(side="left")

        self.training_figure = Figure(figsize=(8.5, 5.4), dpi=100)
        self.ax_train_loss = self.training_figure.add_subplot(211)
        self.ax_train_acc = self.training_figure.add_subplot(212, sharex=self.ax_train_loss)
        self.train_loss_line = self.ax_train_loss.plot([], [], label="train loss")[0]
        self.val_loss_line = self.ax_train_loss.plot([], [], label="validation loss")[0]
        self.train_acc_line = self.ax_train_acc.plot([], [], label="train accuracy")[0]
        self.val_acc_line = self.ax_train_acc.plot([], [], label="validation accuracy")[0]
        self.ax_train_loss.set_ylabel("Log loss")
        self.ax_train_acc.set_ylabel("Accuracy")
        self.ax_train_acc.set_xlabel("Epoch")
        self.ax_train_acc.set_ylim(-0.02, 1.02)
        self.ax_train_loss.grid(True, alpha=0.25)
        self.ax_train_acc.grid(True, alpha=0.25)
        self.ax_train_loss.legend(loc="best")
        self.ax_train_acc.legend(loc="best")
        self.training_figure.tight_layout()
        self.training_canvas = FigureCanvasTkAgg(self.training_figure, master=self.curves_tab)
        self.training_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.training_canvas.draw_idle()

    def _build_deploy_tab(self) -> None:
        # Deployment ----------------------------------------------------------
        deploy = ttk.LabelFrame(self.deploy_tab, text="1. FFat model deployment", padding=10)
        deploy.pack(fill="x", pady=(0, 9))
        self.upload_btn = ttk.Button(deploy, text="Deploy to device", command=self._upload, state="disabled")
        self.upload_btn.grid(row=0, column=0, padx=(0, 8))
        self.save_model_btn = ttk.Button(
            deploy, text="Save .nai package", command=self._save_model, state="disabled"
        )
        self.save_model_btn.grid(row=0, column=1, padx=(0, 8))
        self.upload_progress = tk.StringVar(value="No model deployed in this session.")
        ttk.Label(deploy, textvariable=self.upload_progress).grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(7, 0)
        )

        # Mode controls -------------------------------------------------------
        mode_box = ttk.LabelFrame(self.deploy_tab, text="2. Device mode", padding=10)
        mode_box.pack(fill="x", pady=(0, 9))
        self.training_btn = ttk.Button(
            mode_box, text="TRAINING MODE", command=self._request_training_mode, state="disabled"
        )
        self.training_btn.grid(row=0, column=0, padx=(0, 8))
        self.inference_btn = ttk.Button(
            mode_box, text="INFERENCE MODE", command=self._request_inference_mode, state="disabled"
        )
        self.inference_btn.grid(row=0, column=1, padx=(0, 18))
        self.mode_var = tk.StringVar(value="Current mode: —")
        ttk.Label(mode_box, textvariable=self.mode_var).grid(row=0, column=2, sticky="w")
        mode_box.columnconfigure(2, weight=1)

        # Prediction ----------------------------------------------------------
        pred = ttk.LabelFrame(self.deploy_tab, text="3. Last prediction", padding=12)
        pred.pack(fill="x", pady=(0, 9))
        self.prediction_label_var = tk.StringVar(value="—")
        self.prediction_conf_var = tk.StringVar(value="Perform a gesture in Inference mode")
        self.prediction_meta_var = tk.StringVar(value="Hold BOOT → gesture → release BOOT")

        ttk.Label(pred, textvariable=self.prediction_label_var, font=("TkDefaultFont", 34, "bold")).pack()
        ttk.Label(pred, textvariable=self.prediction_conf_var, font=("TkDefaultFont", 12)).pack(pady=(2, 0))
        ttk.Label(pred, textvariable=self.prediction_meta_var).pack(pady=(3, 0))

        # Status log ----------------------------------------------------------
        log = ttk.LabelFrame(self.deploy_tab, text="Status / diagnostics", padding=8)
        log.pack(fill="both", expand=True)
        self.status_text = tk.Text(log, height=10, wrap="word", state="disabled")
        scrollbar = ttk.Scrollbar(log, orient="vertical", command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=scrollbar.set)
        self.status_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self._log("Ready. NAI4 retains raw six-axis gestures and supports raw, quaternion, and estimated-velocity representations.")

    # -------------------------------------------------------------- Plotting --
    def _clear_plot(self) -> None:
        self.live_t.clear()
        for q in self.live_channels:
            q.clear()
        self._plot_dirty = True

    def _append_live_sample(self, t_ms: int, values: tuple[float, ...]) -> None:
        t_s = float(t_ms) / 1000.0
        self.live_t.append(t_s)
        for q, v in zip(self.live_channels, values):
            q.append(float(v))
        self._plot_dirty = True

    def _refresh_plot(self) -> None:
        if self._plot_dirty and self.live_t:
            t = np.asarray(self.live_t, dtype=np.float64)
            ys = [np.asarray(q, dtype=np.float64) for q in self.live_channels]

            # Show only the requested live horizon even if deques contain more.
            cutoff = t[-1] - LIVE_SECONDS
            start = int(np.searchsorted(t, cutoff, side="left"))
            tx = t[start:]

            for line, y in zip(self.accel_lines, ys[:3]):
                line.set_data(tx, y[start:])
            for line, y in zip(self.gyro_lines, ys[3:]):
                line.set_data(tx, y[start:])

            if len(tx) >= 2:
                self.ax_accel.set_xlim(tx[0], tx[-1])
                self.ax_gyro.set_xlim(tx[0], tx[-1])

            self._autoscale_y(self.ax_accel, ys[:3], start, fallback=(-1.2, 1.2))
            self._autoscale_y(self.ax_gyro, ys[3:], start, fallback=(-100.0, 100.0))
            self.canvas.draw_idle()
            self._plot_dirty = False

        self.after(PLOT_REFRESH_MS, self._refresh_plot)

    @staticmethod
    def _autoscale_y(axis, arrays: list[np.ndarray], start: int, fallback: tuple[float, float]) -> None:
        chunks = [a[start:] for a in arrays if len(a) > start]
        if not chunks:
            axis.set_ylim(*fallback)
            return
        values = np.concatenate(chunks)
        values = values[np.isfinite(values)]
        if values.size == 0:
            axis.set_ylim(*fallback)
            return
        lo = float(np.min(values))
        hi = float(np.max(values))
        span = hi - lo
        if span < 1e-6:
            pad = max(abs(lo) * 0.1, 0.1)
        else:
            pad = 0.12 * span
        axis.set_ylim(lo - pad, hi + pad)

    # --------------------------------------------------------------- Helpers --
    def _log(self, text: str) -> None:
        self.status_text.configure(state="normal")
        self.status_text.insert("end", text.rstrip() + "\n")
        self.status_text.see("end")
        self.status_text.configure(state="disabled")

    def _selected_representation(self) -> str:
        label = self.representation_var.get() if hasattr(self, "representation_var") else "Accelerometer"
        return normalize_representation(REPRESENTATION_LABELS.get(label, "accel"))

    def _representation_changed(self) -> None:
        self.model_package = None
        self.model = None
        self.scaler = None
        if hasattr(self, "save_model_btn"):
            self.save_model_btn.configure(state="disabled")
        if hasattr(self, "upload_btn"):
            self.upload_btn.configure(state="disabled")
        self._update_input_dim()

    def _project_legacy_full_features(self, X_full: np.ndarray, representation: str) -> np.ndarray:
        if representation not in RAW_REPRESENTATIONS:
            raise ValueError(
                "This legacy dataset does not contain raw IMU gestures. Quaternion/velocity "
                "representations require a dataset recorded or saved by the NAI4 GUI."
            )
        X_full = np.asarray(X_full, dtype=np.float32)
        n = int(self.window_var.get())
        if X_full.ndim != 2 or X_full.shape[1] != n * FEATURE_CHANNELS:
            raise ValueError(f"Expected legacy six-axis gestures with {n*FEATURE_CHANNELS} features")
        idx = {
            "accel": (0, 1, 2),
            "gyro": (3, 4, 5),
            "accel+gyro": (0, 1, 2, 3, 4, 5),
        }[representation]
        shaped = X_full.reshape(len(X_full), n, FEATURE_CHANNELS)
        return np.asarray(shaped[:, :, idx], dtype=np.float32).reshape(len(X_full), n * len(idx))

    def _build_dataset_representation(self, representation: str) -> np.ndarray:
        n = int(self.window_var.get())
        if len(self.raw_samples) == len(self.targets) and self.raw_samples:
            return np.stack([
                build_representation(raw, n, representation, sample_rate_hz=50)
                for raw in self.raw_samples
            ]).astype(np.float32)
        X_full = np.stack(self.samples).astype(np.float32)
        return self._project_legacy_full_features(X_full, representation)

    def _update_input_dim(self) -> None:
        try:
            n = int(self.window_var.get())
            rep = self._selected_representation()
            channels = representation_channel_count(rep)
            names = REPRESENTATION_CHANNEL_NAMES[rep]
            self.input_dim_var.set(
                f"= {channels*n} MLP inputs ({n} normalized points × {channels} channels: {names})"
            )
            self._update_topology()
        except Exception:
            self.input_dim_var.set("= —")

    def _toggle_connection(self) -> None:
        if self.ble.connected:
            self.ble.disconnect()
        else:
            self.connect_btn.configure(state="disabled")
            self.device_status.set("Scanning...")
            self.ble.connect()

    def _request_training_mode(self) -> None:
        self.ble.set_training()
        self.tabs.select(self.dataset_tab)

    def _request_inference_mode(self) -> None:
        self.ble.set_inference()
        self.tabs.select(self.deploy_tab)

    # --------------------------------------------------------------- Dataset --
    def _add_label(self) -> None:
        if self.setup_locked:
            return
        label = self.label_entry.get().strip()
        if not label:
            return
        if label in self.labels:
            messagebox.showwarning("Duplicate label", "That label already exists.")
            return
        if len(label.encode("utf-8")) > 31:
            messagebox.showwarning("Label too long", "Labels are limited to 31 UTF-8 bytes.")
            return
        if len(self.labels) >= 10:
            messagebox.showwarning("Too many labels", "NoodleAI supports up to 10 labels.")
            return
        self.labels.append(label)
        self.label_list.insert("end", label)
        self.label_entry.delete(0, "end")

    def _remove_label(self) -> None:
        if self.setup_locked:
            return
        sel = self.label_list.curselection()
        if not sel:
            return
        idx = int(sel[0])
        del self.labels[idx]
        self.label_list.delete(idx)

    def _lock_setup(self) -> None:
        if self.setup_locked:
            return
        try:
            normalized_length = int(self.window_var.get())
        except Exception:
            messagebox.showerror("Normalized length", "Enter a valid normalized gesture length.")
            return
        if not 5 <= normalized_length <= 500:
            messagebox.showerror("Normalized length", "Use 5..500 normalized gesture points.")
            return
        if len(self.labels) < 2:
            messagebox.showerror("Labels", "Define at least two labels first.")
            return

        self.setup_locked = True
        self.window_spin.configure(state="disabled")
        self.label_entry.configure(state="disabled")
        self.add_label_btn.configure(state="disabled")
        self.remove_label_btn.configure(state="disabled")
        self.lock_btn.configure(state="disabled")
        self.record_label_combo.configure(values=self.labels)
        self.record_label_var.set(self.labels[0])
        self.record_btn.configure(state="normal" if self.ble.connected else "disabled")
        self.train_btn.configure(state="normal")
        self._update_dataset_counts()
        self._update_topology()
        self._log(
            f"Dataset locked: raw six-axis gestures retained; model representations are "
            f"temporally normalized to {normalized_length} points."
        )

    def _reset_dataset(self) -> None:
        if (self.samples or self.setup_locked) and not messagebox.askyesno(
            "Reset dataset", "Clear all recorded samples and unlock the setup?"
        ):
            return
        self._reset_dataset_silent()
        self.record_label_combo.configure(values=[])
        self.record_label_var.set("")
        self.record_btn.configure(state="disabled")
        self.save_sample_btn.configure(state="disabled")
        self.discard_btn.configure(state="disabled")
        self.train_btn.configure(state="disabled")
        self.upload_btn.configure(state="disabled")
        self.save_model_btn.configure(state="disabled")
        self.dataset_counts.set("Dataset: define labels, then lock setup")
        self.topology_var.set("Topology: —")
        self.train_result.set("")
        self.record_progress.set("Select a label, then hold BOOT to record; release BOOT to finish.")

    def _reset_dataset_silent(self) -> None:
        self.labels = []
        self.samples = []
        self.raw_samples = []
        self.targets = []
        self.raw_lengths = []
        self.durations_ms = []
        self.setup_locked = False
        self.recording = None
        self.pending_window = None
        self.pending_raw = None
        self.pending_label_index = None
        self.pending_raw_length = None
        self.pending_duration_ms = None
        self.pending_gesture_end = None
        self.model_package = None
        self.model = None
        self.scaler = None
        self.train_epochs = []
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        if hasattr(self, "curves_summary_var"):
            self.curves_summary_var.set("Train a model to see loss and accuracy history.")
        if hasattr(self, "train_loss_line"):
            self._update_training_plot()
        self.label_list.delete(0, "end")
        self.window_spin.configure(state="normal")
        self.label_entry.configure(state="normal")
        self.add_label_btn.configure(state="normal")
        self.remove_label_btn.configure(state="normal")
        self.lock_btn.configure(state="normal")

    def _start_record(self) -> None:
        if not self.setup_locked or not self.ble.connected:
            messagebox.showerror("Record", "Connect the device and lock the dataset setup first.")
            return
        if self.record_label_var.get() not in self.labels:
            messagebox.showerror("Record", "Choose a label.")
            return
        self.ble.set_training()
        self.record_progress.set(
            f"Ready for '{self.record_label_var.get()}': hold BOOT, perform the gesture, release BOOT."
        )

    def _begin_gesture_capture(self) -> None:
        if self.device_mode != "T":
            return
        if not self.setup_locked:
            self._log("Gesture started, but dataset setup is not locked.")
            return
        label = self.record_label_var.get()
        if label not in self.labels:
            self._log("Gesture started, but no valid training label is selected.")
            return

        self.pending_window = None
        self.pending_raw = None
        self.pending_label_index = self.labels.index(label)
        self.pending_raw_length = None
        self.pending_duration_ms = None
        self.pending_gesture_end = None
        self.recording = []
        self.record_progress.set(f"Recording '{label}' while BOOT is held...")
        self.save_sample_btn.configure(state="disabled")
        self.discard_btn.configure(state="normal")

    def _request_finish_gesture(self, expected_count: int, duration_ms: int) -> None:
        if self.device_mode != "T" or self.recording is None:
            return
        self.pending_gesture_end = (expected_count, duration_ms)
        self._finish_gesture_if_ready()

    def _finish_gesture_if_ready(self) -> None:
        if self.recording is None or self.pending_gesture_end is None:
            return
        expected_count, duration_ms = self.pending_gesture_end
        if len(self.recording) < expected_count:
            self.record_progress.set(
                f"BOOT released; waiting for final BLE samples ({len(self.recording)}/{expected_count})..."
            )
            return

        raw = np.asarray(self.recording[:expected_count], dtype=np.float32)
        self.recording = None
        self.pending_gesture_end = None
        if len(raw) < 2:
            self.pending_window = None
            self.pending_label_index = None
            self.record_progress.set("Gesture too short. Hold BOOT and try again.")
            self.discard_btn.configure(state="disabled")
            return

        target_n = int(self.window_var.get())
        self.pending_raw = raw.copy()
        try:
            # Legacy normalized six-axis view; NAI4 feature construction uses raw.
            self.pending_window = preprocess_gesture(raw, target_n)
        except Exception as exc:
            self.pending_window = None
            self.pending_raw = None
            self.pending_label_index = None
            self.record_progress.set(f"Gesture preprocessing failed: {exc}")
            self.discard_btn.configure(state="disabled")
            return

        self.pending_raw_length = int(len(raw))
        self.pending_duration_ms = int(duration_ms)
        self.record_progress.set(
            f"Captured {len(raw)} raw samples ({duration_ms/1000.0:.2f} s) → "
            f"raw six-axis retained; representations normalize to {target_n} points. Save or discard."
        )
        self.save_sample_btn.configure(state="normal")
        self.discard_btn.configure(state="normal")

    def _save_sample(self) -> None:
        if self.pending_window is None or self.pending_raw is None or self.pending_label_index is None:
            return
        self.samples.append(self.pending_window.copy())
        self.raw_samples.append(self.pending_raw.copy())
        self.targets.append(int(self.pending_label_index))
        self.raw_lengths.append(int(self.pending_raw_length or self.window_var.get()))
        self.durations_ms.append(int(self.pending_duration_ms or 0))
        label = self.labels[self.pending_label_index]
        self._log(
            f"Saved '{label}': raw={self.pending_raw_length} samples, "
            f"normalized={self.window_var.get()} × 6."
        )
        self.pending_window = None
        self.pending_raw = None
        self.pending_label_index = None
        self.pending_raw_length = None
        self.pending_duration_ms = None
        self.pending_gesture_end = None
        self.recording = None
        self.record_progress.set("Saved. Select a label and hold BOOT for another gesture.")
        self.save_sample_btn.configure(state="disabled")
        self.discard_btn.configure(state="disabled")
        self.record_btn.configure(state="normal" if self.ble.connected else "disabled")
        self.model_package = None
        self.upload_btn.configure(state="disabled")
        self.save_model_btn.configure(state="disabled")
        self._update_dataset_counts()

    def _discard_pending(self) -> None:
        self.recording = None
        self.pending_window = None
        self.pending_raw = None
        self.pending_label_index = None
        self.pending_raw_length = None
        self.pending_duration_ms = None
        self.pending_gesture_end = None
        self.record_progress.set("Gesture discarded. Hold BOOT to record another.")
        self.save_sample_btn.configure(state="disabled")
        self.discard_btn.configure(state="disabled")
        self.record_btn.configure(state="normal" if self.setup_locked and self.ble.connected else "disabled")

    def _update_dataset_counts(self) -> None:
        if not self.setup_locked:
            return
        counts = [self.targets.count(i) for i in range(len(self.labels))]
        self.dataset_counts.set(
            "Dataset: " + " | ".join(f"{label}: {count}" for label, count in zip(self.labels, counts))
        )

    def _parse_hidden(self) -> tuple[int, ...]:
        text = self.hidden_var.get().strip()
        if not text:
            return tuple()
        try:
            hidden = tuple(int(x.strip()) for x in text.split(",") if x.strip())
        except ValueError as exc:
            raise ValueError("Hidden layers must be comma-separated integers, e.g. 32,16") from exc
        if len(hidden) > 7 or any(x < 1 or x > 512 for x in hidden):
            raise ValueError("Use at most 7 hidden layers, each with 1..512 neurons")
        return hidden

    def _update_topology(self) -> None:
        if not hasattr(self, "topology_var") or not self.setup_locked:
            return
        try:
            hidden = self._parse_hidden()
            rep = self._selected_representation()
            channels = representation_channel_count(rep)
            dims = [int(self.window_var.get()) * channels, *hidden, len(self.labels)]
            self.topology_var.set(
                "Topology: " + " → ".join(map(str, dims)) + f"   [{rep}]"
            )
        except Exception:
            self.topology_var.set("Topology: invalid hidden-layer list")

    def _update_training_plot(self) -> None:
        if not hasattr(self, "training_canvas"):
            return
        if not self.train_epochs:
            for line in (self.train_loss_line, self.val_loss_line, self.train_acc_line, self.val_acc_line):
                line.set_data([], [])
            self.ax_train_loss.set_xlim(0, 1)
            self.ax_train_loss.set_ylim(0, 1)
            self.ax_train_acc.set_xlim(0, 1)
            self.ax_train_acc.set_ylim(-0.02, 1.02)
            self.training_canvas.draw_idle()
            return

        x = np.asarray(self.train_epochs, dtype=np.float64)
        train_loss = np.asarray(self.train_loss_history, dtype=np.float64)
        val_loss = np.asarray(self.val_loss_history, dtype=np.float64)
        train_acc = np.asarray(self.train_acc_history, dtype=np.float64)
        val_acc = np.asarray(self.val_acc_history, dtype=np.float64)

        self.train_loss_line.set_data(x, train_loss)
        self.val_loss_line.set_data(x, val_loss)
        self.train_acc_line.set_data(x, train_acc)
        self.val_acc_line.set_data(x, val_acc)

        self.ax_train_loss.relim()
        self.ax_train_loss.autoscale_view()
        self.ax_train_acc.set_xlim(max(1.0, float(x[0])), max(2.0, float(x[-1])))
        self.ax_train_acc.set_ylim(-0.02, 1.02)
        self.training_canvas.draw_idle()

    def _training_progress(
        self, epoch: int, total_epochs: int, train_loss: float, val_loss: float, val_acc: float
    ) -> None:
        self.train_result.set(
            f"epoch {epoch}/{total_epochs} | loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | val acc {100*val_acc:.1f}%"
        )

    # --------------------------------------------------------------- Training --
    def _export_package(self, clf, scaler, normalized_length: int, representation: str):
        return export_sklearn_mlp(
            clf, scaler, self.labels, normalized_length, representation=representation
        )

    def _train(self) -> None:
        if not self.setup_locked:
            return
        self._update_topology()
        try:
            hidden = self._parse_hidden()
        except ValueError as exc:
            messagebox.showerror("Hidden layers", str(exc))
            return
        if not hidden:
            messagebox.showerror("Hidden layers", "Define at least one hidden layer.")
            return
        try:
            epochs = int(self.epochs_var.get())
        except Exception:
            messagebox.showerror("Epochs", "Enter a valid number of epochs.")
            return
        if not 10 <= epochs <= 5000:
            messagebox.showerror("Epochs", "Use 10..5000 epochs.")
            return

        counts = [self.targets.count(i) for i in range(len(self.labels))]
        if any(c < 5 for c in counts):
            messagebox.showerror(
                "More examples needed",
                "Record at least 5 examples for every label before training. More is strongly recommended.",
            )
            return

        y = np.asarray(self.targets, dtype=np.int32)
        representation = self._selected_representation()
        try:
            X = self._build_dataset_representation(representation)
        except Exception as exc:
            messagebox.showerror("Dataset representation", str(exc))
            return

        self.train_epochs = []
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self._update_training_plot()
        self.curves_summary_var.set("Training in progress...")
        self.train_btn.configure(state="disabled")
        self.train_result.set("Training...")
        self._log(
            f"Training {representation} MLP: X={X.shape}, hidden={hidden}, epochs={epochs}."
        )

        def worker() -> None:
            try:
                n_classes = len(self.labels)
                classes = np.arange(n_classes, dtype=np.int32)
                test_count = max(n_classes, int(round(0.20 * len(y))))
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_count, random_state=42, stratify=y
                )
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                # partial_fit preserves the optimizer state and lets us measure
                # a genuine TensorFlow-style history after every epoch.
                clf = MLPClassifier(
                    hidden_layer_sizes=hidden,
                    activation="relu",
                    solver="adam",
                    random_state=42,
                    max_iter=1,
                    warm_start=False,
                )

                epoch_axis: list[int] = []
                train_losses: list[float] = []
                val_losses: list[float] = []
                train_accs: list[float] = []
                val_accs: list[float] = []

                for epoch in range(1, epochs + 1):
                    if epoch == 1:
                        clf.partial_fit(X_train_s, y_train, classes=classes)
                    else:
                        clf.partial_fit(X_train_s, y_train)

                    train_prob = clf.predict_proba(X_train_s)
                    val_prob = clf.predict_proba(X_test_s)
                    train_pred = np.argmax(train_prob, axis=1)
                    val_pred = np.argmax(val_prob, axis=1)
                    tr_loss = float(log_loss(y_train, train_prob, labels=classes))
                    va_loss = float(log_loss(y_test, val_prob, labels=classes))
                    tr_acc = float(accuracy_score(y_train, train_pred))
                    va_acc = float(accuracy_score(y_test, val_pred))

                    epoch_axis.append(epoch)
                    train_losses.append(tr_loss)
                    val_losses.append(va_loss)
                    train_accs.append(tr_acc)
                    val_accs.append(va_acc)

                    if epoch == 1 or epoch == epochs or epoch % TRAIN_PROGRESS_EVERY == 0:
                        self.after(
                            0,
                            lambda e=epoch, tl=tr_loss, vl=va_loss, va=va_acc: self._training_progress(
                                e, epochs, tl, vl, va
                            ),
                        )

                train_acc = train_accs[-1]
                val_acc = val_accs[-1]
                package = self._export_package(clf, scaler, int(self.window_var.get()), representation)

                ref = NoodleAIReference.from_package(package)
                export_pred = ref.predict(X_test)
                sklearn_pred = clf.predict(X_test_s)
                export_match = accuracy_score(sklearn_pred, export_pred)

                self.after(
                    0,
                    lambda: self._training_done(
                        clf, scaler, package, train_acc, val_acc, export_match,
                        epoch_axis, train_losses, val_losses, train_accs, val_accs,
                        len(y_train), len(y_test), representation
                    ),
                )
            except Exception as exc:
                self.after(0, lambda text=str(exc): self._training_failed(text))

        threading.Thread(target=worker, daemon=True).start()

    def _training_done(
        self,
        clf: MLPClassifier,
        scaler: StandardScaler,
        package: NoodleAIFilePackage,
        train_acc: float,
        val_acc: float,
        export_match: float,
        epoch_axis: list[int],
        train_losses: list[float],
        val_losses: list[float],
        train_accs: list[float],
        val_accs: list[float],
        n_train: int,
        n_val: int,
        representation: str,
    ) -> None:
        self.model = clf
        self.scaler = scaler
        self.model_package = package
        self.train_epochs = epoch_axis
        self.train_loss_history = train_losses
        self.val_loss_history = val_losses
        self.train_acc_history = train_accs
        self.val_acc_history = val_accs
        self._update_training_plot()
        self.curves_summary_var.set(
            f"Representation: {representation} | Train: {n_train} | Validation: {n_val} | "
            f"Epochs: {len(epoch_axis)} | Final val loss: {val_losses[-1]:.4f}"
        )
        self.train_result.set(
            f"train {100*train_acc:.1f}% | validation {100*val_acc:.1f}% | "
            f"export match {100*export_match:.1f}%"
        )
        self._log(
            f"Training complete: representation={representation}, train={n_train}, validation={n_val}, "
            f"epochs={len(epoch_axis)}, model={package.total_bytes/1024:.1f} KiB."
        )
        self.train_btn.configure(state="normal")
        self.save_model_btn.configure(state="normal")
        self.upload_btn.configure(state="normal" if self.ble.connected else "disabled")
        self.tabs.select(self.curves_tab)

    def _training_failed(self, text: str) -> None:
        self.train_btn.configure(state="normal")
        self.train_result.set("Training failed")
        self._log("ERROR: " + text)
        messagebox.showerror("Training failed", text)

    # -------------------------------------------------------------- Deployment --
    def _upload(self) -> None:
        if self.model_package is None:
            return
        if not self.ble.connected:
            messagebox.showerror("Upload", "Connect the device first.")
            return
        self.upload_progress.set(f"Deploying {self.model_package.total_bytes/1024:.1f} KiB to FFat...")
        self.upload_btn.configure(state="disabled")
        self.ble.upload_package(self.model_package)

    def _save_model(self) -> None:
        if self.model_package is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save NoodleAI model",
            defaultextension=".nai",
            filetypes=[("NoodleAI model", "*.nai"), ("All files", "*")],
        )
        if path:
            Path(path).write_bytes(self.model_package.to_archive_bytes())
            self._log(f"Saved model: {path}")

    # --------------------------------------------------------------- Save/load --
    def _save_dataset(self) -> None:
        if not self.setup_locked or not self.samples:
            messagebox.showinfo("Save dataset", "Record at least one sample first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save six-axis IMU dataset",
            defaultextension=".npz",
            filetypes=[("NumPy dataset", "*.npz")],
        )
        if not path:
            return
        normalized_length = int(self.window_var.get())
        arrays = {
            "X": np.stack(self.samples).astype(np.float32),
            "y": np.asarray(self.targets, dtype=np.int32),
            "labels": np.asarray(self.labels),
            "normalized_length": np.int32(normalized_length),
            "window_length": np.int32(normalized_length),
            "raw_lengths": np.asarray(self.raw_lengths, dtype=np.int32),
            "durations_ms": np.asarray(self.durations_ms, dtype=np.int32),
            "sample_rate_hz": np.int32(50),
            "channels": np.asarray(FEATURE_NAMES),
            "channel_count": np.int32(FEATURE_CHANNELS),
            "preprocess": np.asarray(GESTURE_PREPROCESS_VERSION),
        }
        if len(self.raw_samples) == len(self.targets) and self.raw_samples:
            offsets = np.zeros(len(self.raw_samples) + 1, dtype=np.int32)
            for i, raw in enumerate(self.raw_samples):
                offsets[i + 1] = offsets[i] + len(raw)
            arrays["raw_offsets"] = offsets
            arrays["raw_data"] = np.concatenate(self.raw_samples, axis=0).astype(np.float32)
        np.savez_compressed(path, **arrays)
        self._log(f"Saved dataset: {path}")

    def _load_dataset(self) -> None:
        if self.samples or self.setup_locked:
            if not messagebox.askyesno("Load dataset", "Replace the current dataset/setup?"):
                return
        path = filedialog.askopenfilename(
            title="Load six-axis IMU dataset", filetypes=[("NumPy dataset", "*.npz")]
        )
        if not path:
            return
        try:
            data = np.load(path, allow_pickle=False)
            X = np.asarray(data["X"], dtype=np.float32)
            y = np.asarray(data["y"], dtype=np.int32)
            labels = [str(x) for x in np.asarray(data["labels"]).tolist()]
            key = "normalized_length" if "normalized_length" in data.files else "window_length"
            normalized_length = int(np.asarray(data[key]).item())
            expected_dim = normalized_length * FEATURE_CHANNELS
            if X.ndim != 2 or X.shape[1] != expected_dim or len(X) != len(y):
                raise ValueError(
                    f"This GUI expects a six-axis dataset with {expected_dim} features "
                    f"({normalized_length} × 6); got X.shape={X.shape}."
                )
            if len(labels) < 2 or np.any(y < 0) or np.any(y >= len(labels)):
                raise ValueError("Dataset labels are invalid")

            raw_lengths = (
                np.asarray(data["raw_lengths"], dtype=np.int32)
                if "raw_lengths" in data.files
                else np.full(len(y), normalized_length, dtype=np.int32)
            )
            if len(raw_lengths) != len(y):
                raise ValueError("Dataset raw-length metadata is invalid")

            loaded_raw_samples: list[np.ndarray] = []
            if "raw_data" in data.files and "raw_offsets" in data.files:
                raw_data = np.asarray(data["raw_data"], dtype=np.float32)
                raw_offsets = np.asarray(data["raw_offsets"], dtype=np.int32)
                if raw_data.ndim != 2 or raw_data.shape[1] != 6 or len(raw_offsets) != len(y) + 1:
                    raise ValueError("NAI4 raw gesture storage is invalid")
                if raw_offsets[0] != 0 or raw_offsets[-1] != len(raw_data) or np.any(np.diff(raw_offsets) < 2):
                    raise ValueError("NAI4 raw gesture offsets are invalid")
                loaded_raw_samples = [
                    raw_data[raw_offsets[i]:raw_offsets[i + 1]].copy() for i in range(len(y))
                ]

            durations_ms = (
                np.asarray(data["durations_ms"], dtype=np.int32)
                if "durations_ms" in data.files
                else np.zeros(len(y), dtype=np.int32)
            )
            if len(durations_ms) != len(y):
                raise ValueError("Dataset duration metadata is invalid")

            self._reset_dataset_silent()
            self.window_var.set(normalized_length)
            self.labels = labels
            for label in labels:
                self.label_list.insert("end", label)
            self.samples = [row.copy() for row in X]
            self.raw_samples = loaded_raw_samples
            self.targets = [int(v) for v in y]
            self.raw_lengths = [int(v) for v in raw_lengths]
            self.durations_ms = [int(v) for v in durations_ms]
            self._lock_setup()
            self._update_dataset_counts()
            raw_note = "raw gestures available" if self.raw_samples else "legacy normalized-only dataset"
            self._log(f"Loaded dataset: {path} ({raw_note})")
        except Exception as exc:
            messagebox.showerror("Load dataset", str(exc))

    # --------------------------------------------------------------- BLE loop --
    def _handle_imu_event(self, event: tuple) -> None:
        # Six-axis live firmware: ("imu", t_ms, ax, ay, az, gx, gy, gz)
        if len(event) >= 8:
            _, t_ms, ax, ay, az, gx, gy, gz = event[:8]
        # Graceful fallback if an older 3-axis BLE helper is accidentally used.
        elif len(event) >= 5:
            _, t_ms, ax, ay, az = event[:5]
            gx = gy = gz = 0.0
        else:
            return

        values = (float(ax), float(ay), float(az), float(gx), float(gy), float(gz))
        self.accel_status.set(f"ax: {ax:+.3f} g    ay: {ay:+.3f} g    az: {az:+.3f} g")
        self.gyro_status.set(f"gx: {gx:+.1f} °/s    gy: {gy:+.1f} °/s    gz: {gz:+.1f} °/s")
        self._append_live_sample(int(t_ms), values)

        if self.recording is not None:
            self.recording.append(list(values))
            self.record_progress.set(f"Recording gesture: {len(self.recording)} raw samples...")
            self._finish_gesture_if_ready()

    def _poll_ble_events(self) -> None:
        try:
            while True:
                event = self.ble.events.get_nowait()
                kind = event[0]

                if kind == "connected":
                    self.device_status.set(f"Connected: {event[1]}")
                    self.connect_btn.configure(text="Disconnect", state="normal")
                    self.inference_btn.configure(state="normal")
                    self.training_btn.configure(state="normal")
                    if self.setup_locked:
                        self.record_btn.configure(state="normal")
                    if self.model_package is not None:
                        self.upload_btn.configure(state="normal")
                    self._log(f"BLE connected to {event[1]}")

                elif kind == "disconnected":
                    self.device_mode = "?"
                    self.mode_var.set("Current mode: —")
                    self.device_status.set("Disconnected")
                    self.connect_btn.configure(text="Connect", state="normal")
                    self.record_btn.configure(state="disabled")
                    self.upload_btn.configure(state="disabled")
                    self.inference_btn.configure(state="disabled")
                    self.training_btn.configure(state="disabled")
                    self.recording = None
                    self.pending_gesture_end = None
                    self._log("BLE disconnected")

                elif kind == "imu":
                    self._handle_imu_event(event)

                elif kind == "status":
                    text = event[1]
                    self._log("Device: " + text)

                    if text == "MODE:T":
                        self.device_mode = "T"
                        self.mode_var.set("Current mode: TRAINING")
                        self.device_status.set("Training mode — select label, then use BOOT")
                        self.record_progress.set("Select a label, hold BOOT to record, release BOOT to finish.")

                    elif text == "MODE:I":
                        self.device_mode = "I"
                        self.mode_var.set("Current mode: INFERENCE")
                        self.device_status.set("Inference mode — hold BOOT to perform a gesture")

                    elif text == "GESTURE:START":
                        if self.device_mode == "T":
                            self._begin_gesture_capture()
                        else:
                            self.device_status.set("Inference: recording gesture...")
                            self.prediction_meta_var.set("Recording gesture...")

                    elif text.startswith("GESTURE:END:"):
                        parts = text.split(":")
                        if len(parts) >= 4:
                            try:
                                raw_count = int(parts[2])
                                duration_ms = int(parts[3])
                                if self.device_mode == "T":
                                    self._request_finish_gesture(raw_count, duration_ms)
                                else:
                                    self.prediction_meta_var.set(
                                        f"raw={raw_count} samples | duration={duration_ms/1000.0:.2f} s"
                                    )
                                    self.device_status.set("Inference: classifying...")
                            except ValueError:
                                pass

                    elif text.startswith("GESTURE:SHORT"):
                        self.recording = None
                        self.pending_gesture_end = None
                        self.pending_window = None
                        self.pending_raw = None
                        self.pending_label_index = None
                        self.pending_raw_length = None
                        self.pending_duration_ms = None
                        self.save_sample_btn.configure(state="disabled")
                        self.discard_btn.configure(state="disabled")
                        self.record_progress.set("Gesture too short. Hold BOOT a little longer and try again.")
                        self.prediction_meta_var.set("Gesture too short — try again")

                    elif text == "MODEL_OK":
                        self.upload_progress.set("FFat model verified, activated, and ready.")
                        self.upload_btn.configure(state="normal" if self.model_package is not None else "disabled")
                        self.tabs.select(self.deploy_tab)

                    elif text.startswith("ERR:"):
                        self.upload_progress.set(text)
                        self.upload_btn.configure(
                            state="normal" if self.model_package is not None and self.ble.connected else "disabled"
                        )

                    elif text.startswith("P:"):
                        parts = text.split(":")
                        if len(parts) >= 3:
                            try:
                                idx = int(parts[1])
                                conf = float(parts[2])
                                name = self.labels[idx] if 0 <= idx < len(self.labels) else str(idx)
                                self.device_status.set(f"Inference: {name} ({100*conf:.1f}%)")
                                self.prediction_label_var.set(name)
                                self.prediction_conf_var.set(f"Confidence {100*conf:.1f}%")
                            except Exception:
                                pass

                elif kind == "upload_progress":
                    sent, total, filename = event[1], event[2], event[3]
                    pct = 100.0 * sent / max(total, 1)
                    self.upload_progress.set(f"Deploying {filename}: {pct:.0f}% ({sent}/{total} bytes)")

                elif kind == "info":
                    self._log(event[1])

                elif kind == "error":
                    self._log("ERROR: " + event[1])
                    self.device_status.set("BLE error")
                    self.connect_btn.configure(text="Connect", state="normal")
                    self.upload_btn.configure(
                        state="normal" if self.model_package is not None and self.ble.connected else "disabled"
                    )

        except queue.Empty:
            pass
        self.after(30, self._poll_ble_events)

    def _on_close(self) -> None:
        if self.ble.connected:
            self.ble.disconnect()
        self.destroy()


if __name__ == "__main__":
    app = NoodleTrainerApp()
    app.mainloop()
