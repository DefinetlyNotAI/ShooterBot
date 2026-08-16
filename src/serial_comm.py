"""Generic serial communication layer for Arduino and similar devices.

Supports simulation mode, auto reconnect, optional CRC, acknowledgements,
and configurable packet formats.
"""

from __future__ import annotations

import binascii
import json
import logging
import threading
import time
from typing import Any, Callable

logger = logging.getLogger("realtime_cv.serial")


class SerialInterface:
    def __init__(
            self,
            port: str = "COM3",
            baudrate: int = 115200,
            crc: bool = True,
            simulation: bool = True,
            ack_timeout: float = 1.0,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.crc = crc
        self.simulation = simulation
        self.ack_timeout = ack_timeout

        self._running = False
        self._lock = threading.Lock()

        self._conn: Any = None
        self._thread: threading.Thread | None = None
        self._on_receive: Callable[[bytes], None] = lambda _data: None

        self._last_sent: dict[int, float] = {}

    def _connect_loop(self) -> None:
        try:
            import serial
        except Exception as exc:
            logger.warning(
                "pyserial not available; switching to simulation mode: %s",
                exc,
            )

            self.simulation = True

            while self._running:
                time.sleep(0.5)

            return

        retry_count = 0

        while self._running:
            try:
                if self.simulation:
                    time.sleep(0.5)
                    continue

                if self._conn is None:
                    logger.info(
                        "Opening serial port %s at %s",
                        self.port,
                        self.baudrate,
                    )

                    try:
                        self._conn = serial.Serial(
                            self.port,
                            self.baudrate,
                            timeout=0.1,
                        )
                        retry_count = 0

                    except Exception as exc:
                        retry_count += 1

                        logger.warning(
                            "Failed to open serial port %s: %s",
                            self.port,
                            exc,
                        )

                        if retry_count >= 3:
                            logger.warning(
                                "Giving up on serial port after %s attempts; "
                                "switching to simulation",
                                retry_count,
                            )
                            self.simulation = True

                        time.sleep(1.0)
                        continue

                connection = self._conn

                if connection is not None and connection.in_waiting:
                    data = connection.read(connection.in_waiting)

                    self._invoke_receive_callback(data)

                time.sleep(0.01)

            except Exception as exc:
                logger.warning(
                    "Serial error: %s",
                    exc,
                )

                connection = self._conn

                if connection is not None:
                    try:
                        connection.close()
                    except Exception:
                        pass

                self._conn = None

                time.sleep(1.0)

    def start(self) -> None:
        if self._running:
            return

        self._running = True

        thread = threading.Thread(
            target=self._connect_loop,
            name="serial-interface",
            daemon=True,
        )

        self._thread = thread
        thread.start()

    def stop(self) -> None:
        self._running = False

        thread = self._thread

        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

        connection = self._conn

        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass

        self._conn = None
        self._thread = None

    def set_receive_callback(
            self,
            callback: Callable[[bytes], None],
    ) -> None:
        self._on_receive = callback

    def _invoke_receive_callback(self, data: bytes) -> None:
        try:
            self._on_receive(data)
        except Exception:
            logger.warning(
                "Serial receive callback failed",
                exc_info=True,
            )

    @staticmethod
    def _compute_crc(payload: bytes) -> bytes:
        """Compute a CRC32 checksum as four big-endian bytes."""
        crc = binascii.crc32(payload) & 0xFFFFFFFF
        return crc.to_bytes(4, "big")

    def send_packet(
            self,
            payload: bytes,
            require_ack: bool = False,
    ) -> bool:
        packet = payload

        if self.crc:
            packet += self._compute_crc(payload)

        if self.simulation:
            logger.info(
                "[SIM] send_packet: %s",
                packet.hex(),
            )
            return True

        with self._lock:
            connection = self._conn

            if connection is None:
                logger.warning(
                    "Cannot send packet: serial connection is not open"
                )
                return False

            try:
                connection.write(packet)

                if not require_ack:
                    return True

                start = time.monotonic()

                while time.monotonic() - start < self.ack_timeout:
                    if connection.in_waiting:
                        response = connection.read(connection.in_waiting)

                        logger.debug(
                            "Received ack: %s",
                            response.hex(),
                        )

                        return True

                    time.sleep(0.01)

                logger.warning(
                    "Serial acknowledgement timed out after %.3f seconds",
                    self.ack_timeout,
                )

                return False

            except Exception as exc:
                logger.warning(
                    "Send error: %s",
                    exc,
                )
                return False

    def send_json(
            self,
            obj: dict[str, Any],
            require_ack: bool = False,
    ) -> bool:
        data = json.dumps(
            obj,
            separators=(",", ":"),
        ).encode("utf-8")

        return self.send_packet(
            data,
            require_ack=require_ack,
        )

    def send_telemetry(
            self,
            object_id: int,
            class_name: str,
            confidence: float,
            normalized_center: list[float],
            velocity: list[float],
            timestamp: float,
            predicted_center: list[float] | None = None,
            require_ack: bool = False,
            advanced: bool = False,
    ) -> bool:
        """Send object telemetry over the serial connection.

        The default packet contains only normalized X/Y coordinates.

        With ``advanced=True``, the packet additionally contains:

        - object ID
        - class name
        - detection confidence
        - X/Y velocity
        - timestamp
        - predicted X/Y coordinates when available

        Telemetry is throttled to approximately 10 Hz per object.
        """

        now = time.monotonic()
        last_sent = self._last_sent.get(object_id)

        if last_sent is not None and now - last_sent < 0.08:
            return True

        self._last_sent[object_id] = now

        try:
            nx = float(normalized_center[0])
            ny = float(normalized_center[1])
        except (IndexError, TypeError, ValueError):
            nx = 0.0
            ny = 0.0

        payload: dict[str, Any] = {
            "x": round(nx, 4),
            "y": round(ny, 4),
        }

        if advanced:
            try:
                vx = float(velocity[0])
                vy = float(velocity[1])
            except (IndexError, TypeError, ValueError):
                vx = 0.0
                vy = 0.0

            payload.update(
                {
                    "id": object_id,
                    "class": class_name,
                    "confidence": round(float(confidence), 4),
                    "vx": round(vx, 4),
                    "vy": round(vy, 4),
                    "timestamp": float(timestamp),
                }
            )

        if advanced and predicted_center is not None:
            try:
                px = float(predicted_center[0])
                py = float(predicted_center[1])

                payload.update(
                    {
                        "px": round(px, 4),
                        "py": round(py, 4),
                    }
                )

            except (IndexError, TypeError, ValueError):
                pass

        sent = self.send_json(
            payload,
            require_ack=require_ack,
        )

        if self.simulation:
            if self._on_receive is not None:
                echo_payload = json.dumps(
                    payload,
                    separators=(",", ":"),
                ).encode("utf-8")

                def echo() -> None:
                    time.sleep(0.01)

                    try:
                        self._invoke_receive_callback(echo_payload)
                    except Exception:
                        logger.debug(
                            "Simulation receive callback failed",
                            exc_info=True,
                        )

                echo_thread = threading.Thread(
                    target=echo,
                    name="serial-simulation-echo",
                    daemon=True,
                )

                echo_thread.start()

        return sent
