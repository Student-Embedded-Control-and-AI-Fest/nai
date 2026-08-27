from __future__ import annotations

import asyncio
import queue
import struct
import threading
import zlib

from bleak import BleakClient, BleakScanner

from noodle_model import NoodleAIFilePackage

DEVICE_PREFIX = "NoodleAI"
SERVICE_UUID = "7f8b0001-5f5b-4f4a-a5d5-2e889aa10001"
IMU_UUID = "7f8b0002-5f5b-4f4a-a5d5-2e889aa10001"
CONTROL_UUID = "7f8b0003-5f5b-4f4a-a5d5-2e889aa10001"
MODEL_UUID = "7f8b0004-5f5b-4f4a-a5d5-2e889aa10001"
STATUS_UUID = "7f8b0005-5f5b-4f4a-a5d5-2e889aa10001"

OP_SET_TRAINING = 0x01
OP_SET_INFERENCE = 0x02
OP_ERASE_MODEL = 0x12
OP_DEPLOY_BEGIN = 0x20
OP_FILE_BEGIN = 0x21
OP_FILE_END = 0x22
OP_DEPLOY_COMMIT = 0x23
OP_DEPLOY_ABORT = 0x24

IMU_BATCH_SIZE = 5
IMU_SAMPLE_RATE_HZ = 50
IMU_SAMPLE_PERIOD_MS = 1000 // IMU_SAMPLE_RATE_HZ
IMU_PACKET = struct.Struct("<IB3x30f")  # t0_ms + count + pad + 5 * 6-axis float samples = 128 bytes


class BLELink:
    """Runs bleak on its own asyncio thread and exposes events to Tkinter."""

    def __init__(self) -> None:
        self.events: queue.Queue[tuple] = queue.Queue()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._client: BleakClient | None = None
        self._connected = False
        self._status_rx: asyncio.Queue[str] | None = None

    @property
    def connected(self) -> bool:
        return self._connected

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def connect(self) -> None:
        self._submit(self._connect())

    def disconnect(self) -> None:
        self._submit(self._disconnect())

    def set_training(self) -> None:
        self._submit(self._write_control(bytes([OP_SET_TRAINING])))

    def set_inference(self) -> None:
        self._submit(self._write_control(bytes([OP_SET_INFERENCE])))

    def erase_model(self) -> None:
        self._submit(self._write_control(bytes([OP_ERASE_MODEL])))

    def upload_package(self, package: NoodleAIFilePackage) -> None:
        self._submit(self._upload_package(package))

    async def _connect(self) -> None:
        if self._connected:
            self.events.put(("info", "Already connected"))
            return
        try:
            self.events.put(("info", "Scanning for NoodleAI..."))
            devices = await BleakScanner.discover(timeout=5.0)
            target = next((d for d in devices if d.name and d.name.startswith(DEVICE_PREFIX)), None)
            if target is None:
                raise RuntimeError("No NoodleAI BLE device found")

            self._client = BleakClient(target, disconnected_callback=self._on_disconnect)
            await self._client.connect()
            self._status_rx = asyncio.Queue()
            await self._client.start_notify(IMU_UUID, self._on_imu)
            await self._client.start_notify(STATUS_UUID, self._on_status)
            self._connected = True
            self.events.put(("connected", target.name or DEVICE_PREFIX))

            # Give BlueZ and the ESP32 CCCDs a moment to settle after the two
            # notification subscriptions before issuing the first control write.
            await asyncio.sleep(0.20)
            await self._write_control(bytes([OP_SET_TRAINING]))
        except Exception as exc:
            self._connected = False
            self._client = None
            self.events.put(("error", f"BLE connect failed: {exc}"))

    async def _disconnect(self) -> None:
        try:
            if self._client and self._client.is_connected:
                await self._client.disconnect()
        finally:
            self._connected = False
            self._client = None
            self.events.put(("disconnected",))

    def _on_disconnect(self, _client) -> None:
        self._connected = False
        self.events.put(("disconnected",))

    def _on_imu(self, _sender, data: bytearray) -> None:
        if len(data) != IMU_PACKET.size:
            self.events.put(("error", f"Unexpected IMU packet size: {len(data)} bytes (expected {IMU_PACKET.size})"))
            return

        values = IMU_PACKET.unpack(data)
        t0_ms = values[0]
        count = int(values[1])
        six = values[2:]
        if not 1 <= count <= IMU_BATCH_SIZE:
            self.events.put(("error", f"Invalid IMU sample count: {count}"))
            return

        # Count-aware batching preserves the final partial packet at BOOT
        # release. The GUI always receives all six channels, even when the
        # deployed model later chooses accel-only or gyro-only.
        for i in range(count):
            j = 6 * i
            ax, ay, az, gx, gy, gz = six[j : j + 6]
            t_ms = (t0_ms + i * IMU_SAMPLE_PERIOD_MS) & 0xFFFFFFFF
            self.events.put(("imu", t_ms, ax, ay, az, gx, gy, gz))

    def _on_status(self, _sender, data: bytearray) -> None:
        text = bytes(data).decode("utf-8", errors="replace")
        self.events.put(("status", text))
        if self._status_rx is not None:
            try:
                self._status_rx.put_nowait(text)
            except asyncio.QueueFull:
                pass

    async def _wait_status(self, expected_prefix: str, timeout: float = 5.0) -> str:
        if self._status_rx is None:
            raise RuntimeError("Status notification queue is not ready")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for device status {expected_prefix!r}")
            text = await asyncio.wait_for(self._status_rx.get(), timeout=remaining)
            if text.startswith("ERR:"):
                raise RuntimeError(f"Device reported {text}")
            if text.startswith(expected_prefix):
                return text

    async def _write_control(self, payload: bytes) -> None:
        if not self._client or not self._client.is_connected:
            self.events.put(("error", "Not connected"))
            return
        await self._client.write_gatt_char(CONTROL_UUID, payload, response=False)

    async def _upload_package(self, package: NoodleAIFilePackage) -> None:
        if not self._client or not self._client.is_connected:
            self.events.put(("error", "Not connected"))
            return

        total = package.total_bytes
        sent_total = 0
        try:
            # Clear stale statuses from earlier mode changes so every deployment
            # step waits for the acknowledgement generated by that step.
            if self._status_rx is not None:
                while not self._status_rx.empty():
                    try:
                        self._status_rx.get_nowait()
                    except asyncio.QueueEmpty:
                        break

            await self._write_control(bytes([OP_DEPLOY_BEGIN]))
            await self._wait_status("DEPLOY:", timeout=5.0)

            char = self._client.services.get_characteristic(MODEL_UUID)
            if char is None:
                raise RuntimeError("Model BLE characteristic not found")

            # Use acknowledged GATT Write Requests for model bytes.  Deployment
            # is occasional, so reliability is more important than raw speed.
            # max_write_without_response_size is also a conservative indication
            # of the negotiated ATT payload available on BlueZ.
            max_payload = int(getattr(char, "max_write_without_response_size", 20) or 20)
            chunk_size = max(20, min(160, max_payload))

            for name, data in package.files.items():
                name_b = name.encode("ascii")
                if len(name_b) > 15:
                    raise ValueError(f"Filename too long for BLE protocol: {name}")

                crc = zlib.crc32(data) & 0xFFFFFFFF
                begin = struct.pack("<BBII", OP_FILE_BEGIN, len(name_b), len(data), crc) + name_b
                await self._write_control(begin)
                await self._wait_status(f"FILE_READY:{name}", timeout=5.0)

                pos = 0
                while pos < len(data):
                    chunk = data[pos : pos + chunk_size]
                    # IMPORTANT: response=True serializes the transfer.  The call
                    # returns only after the ESP32 has accepted this chunk.
                    await self._client.write_gatt_char(MODEL_UUID, chunk, response=True)
                    pos += len(chunk)
                    sent_total += len(chunk)
                    self.events.put(("upload_progress", sent_total, total, name))

                await self._write_control(bytes([OP_FILE_END]))
                await self._wait_status(f"FILE_OK:{name}", timeout=8.0)

            await self._write_control(bytes([OP_DEPLOY_COMMIT]))
            self.events.put(("info", "FFat files sent; waiting for Noodle model validation..."))
            await self._wait_status("MODEL_OK", timeout=15.0)
            self.events.put(("info", "Model deployed and validated by Noodle."))

        except Exception as exc:
            try:
                if self._client and self._client.is_connected:
                    await self._client.write_gatt_char(CONTROL_UUID, bytes([OP_DEPLOY_ABORT]), response=False)
            except Exception:
                pass
            self.events.put(("error", f"Model deployment failed: {exc}"))
