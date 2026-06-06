from __future__ import annotations

import signal
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np
from simple_tcp_server import SimpleTcpServer


HOST = "169.254.83.86"
PORT = 8888
WIDTH = 512
HEIGHT = 512
PIXEL_COUNT = WIDTH * HEIGHT
MAGIC = b"DINOIMG1"
HEADER_STRUCT = struct.Struct("<8siiQdii")
DEPTH_MIN_MM = 1
DEPTH_MAX_MM = 4090
WINDOW_NAME = "DINO Raw Sensor Stream"
SHUTDOWN_MESSAGE = "Shutdown requested. Closing raw sensor server..."
VISUALIZATION_INTERVAL_SECONDS = 1.0 / 30.0


@dataclass
class SensorFrame:
    depth: np.ndarray
    infrared: np.ndarray
    sequence: int
    client_timestamp: float
    received_timestamp: float


@dataclass
class SharedState:
    lock: threading.Condition = field(default_factory=threading.Condition)
    latest_frame: SensorFrame | None = None
    receive_fps: float = 0.0
    last_error: str = ""


class RawSensorTcpServer(SimpleTcpServer):
    def _handle(self, conn, addr) -> None:
        connected = datetime.now()
        print(f"[{connected:%Y-%m-%d %H:%M:%S.%f}] Client connected from {addr[0]}:{addr[1]}", flush=True)
        super()._handle(conn, addr)

    def _conn_close(self, addr) -> None:
        print(f"Client disconnected from {addr[0]}:{addr[1]}", flush=True)
        super()._conn_close(addr)


class FpsCounter:
    def __init__(self, sample_seconds: float = 1.5) -> None:
        self.sample_seconds = sample_seconds
        self.samples: deque[float] = deque()

    def tick(self, timestamp: float) -> float:
        self.samples.append(timestamp)
        cutoff = timestamp - self.sample_seconds
        while self.samples and self.samples[0] < cutoff:
            self.samples.popleft()

        if len(self.samples) < 2:
            return 0.0

        elapsed = self.samples[-1] - self.samples[0]
        if elapsed <= 0.0:
            return 0.0

        return (len(self.samples) - 1) / elapsed


def copy_sensor_frame(frame: SensorFrame) -> SensorFrame:
    return SensorFrame(
        depth=frame.depth.copy(),
        infrared=frame.infrared.copy(),
        sequence=frame.sequence,
        client_timestamp=frame.client_timestamp,
        received_timestamp=frame.received_timestamp,
    )


def parse_sensor_images(message: bytes) -> tuple[np.ndarray, np.ndarray, int, float]:
    if len(message) < HEADER_STRUCT.size:
        raise ValueError(f"Message is too short: {len(message)} bytes")

    magic, width, height, sequence, client_timestamp, depth_count, infrared_count = HEADER_STRUCT.unpack_from(message, 0)
    if magic != MAGIC:
        raise ValueError(f"Bad packet magic: {magic!r}")
    if width != WIDTH or height != HEIGHT:
        raise ValueError(f"Unexpected image size: {width}x{height}")
    if depth_count != PIXEL_COUNT or infrared_count != PIXEL_COUNT:
        raise ValueError(f"Unexpected pixel counts: depth={depth_count}, infrared={infrared_count}")

    depth_bytes = depth_count * np.dtype(np.uint16).itemsize
    infrared_bytes = infrared_count * np.dtype(np.uint16).itemsize
    expected_size = HEADER_STRUCT.size + depth_bytes + infrared_bytes
    if len(message) != expected_size:
        raise ValueError(f"Unexpected packet size: got {len(message)}, expected {expected_size}")

    offset = HEADER_STRUCT.size
    depth = np.frombuffer(message, dtype="<u2", count=depth_count, offset=offset).reshape((height, width))
    offset += depth_bytes
    infrared = np.frombuffer(message, dtype="<u2", count=infrared_count, offset=offset).reshape((height, width))

    return depth, infrared, sequence, client_timestamp


def normalize_depth_for_display(depth: np.ndarray) -> np.ndarray:
    clipped = np.clip(depth, DEPTH_MIN_MM, DEPTH_MAX_MM).astype(np.float32, copy=False)
    normalized = (DEPTH_MAX_MM - clipped) * (255.0 / (DEPTH_MAX_MM - DEPTH_MIN_MM))
    display = normalized.astype(np.uint8)
    display[depth == 0] = 0
    return display


def normalize_min_max_for_display(image: np.ndarray) -> np.ndarray:
    min_value = int(image.min())
    max_value = int(image.max())
    if max_value <= min_value:
        return np.zeros(image.shape, dtype=np.uint8)

    scale = 255.0 / float(max_value - min_value)
    return ((image.astype(np.float32) - min_value) * scale).clip(0, 255).astype(np.uint8)


def draw_text(image: np.ndarray, lines: list[str]) -> None:
    y = 24
    for line in lines:
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24


def render_frame(frame: SensorFrame, receive_fps: float) -> np.ndarray:
    depth_display = cv2.cvtColor(normalize_depth_for_display(frame.depth), cv2.COLOR_GRAY2BGR)
    infrared_display = cv2.cvtColor(normalize_min_max_for_display(frame.infrared), cv2.COLOR_GRAY2BGR)

    received_text = datetime.fromtimestamp(frame.received_timestamp).strftime("%H:%M:%S.%f")[:-3]
    draw_text(
        depth_display,
        [
            "Depth 16-bit",
            f"Frame {frame.sequence}",
            f"RX FPS {receive_fps:.1f}",
        ],
    )
    draw_text(
        infrared_display,
        [
            "Infrared 16-bit",
            f"Received {received_text}",
            f"Client TS {frame.client_timestamp:.3f}",
        ],
    )

    return np.hstack((depth_display, infrared_display))


def render_waiting_frame(message: str) -> np.ndarray:
    image = np.zeros((HEIGHT, WIDTH * 2, 3), dtype=np.uint8)
    draw_text(image, ["Waiting for raw sensor frames", message])
    return image


def process_message(message: bytes, state: SharedState, fps_counter: FpsCounter) -> bytes:
    if message == b"quit":
        return b"quit"

    depth, infrared, sequence, client_timestamp = parse_sensor_images(message)
    received_timestamp = time.time()
    receive_fps = fps_counter.tick(time.perf_counter())
    frame = SensorFrame(depth, infrared, sequence, client_timestamp, received_timestamp)

    with state.lock:
        state.latest_frame = frame
        state.receive_fps = receive_fps
        state.last_error = ""
        state.lock.notify_all()

    received = datetime.fromtimestamp(received_timestamp)
    print(
        f"[{received:%Y-%m-%d %H:%M:%S.%f}] "
        f"frame={sequence} client_ts={client_timestamp:.6f} "
        f"rx_fps={receive_fps:.2f} depth_shape={depth.shape} infrared_shape={infrared.shape}",
        flush=True,
    )

    return b"ok"


class RawSensorImageReceiver:
    def __init__(
        self,
        host: str = HOST,
        port: int = PORT,
        client_timeout: float = 3600.0,
        print_frame_log: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.client_timeout = client_timeout
        self.print_frame_log = print_frame_log
        self.state = SharedState()
        self.stop_event = threading.Event()
        self.server_thread: threading.Thread | None = None
        self.server: RawSensorTcpServer | None = None
        self.fps_counter = FpsCounter()

    def start(self) -> None:
        if self.server_thread is not None and self.server_thread.is_alive():
            return

        self.stop_event.clear()
        self.server_thread = threading.Thread(target=self._server_loop, name="RawSensorTcpServer", daemon=True)
        self.server_thread.start()

    def stop(self, join_timeout: float = 1.0) -> None:
        self.stop_event.set()
        if self.server is not None:
            self.server.running = False

        if self.server_thread is not None:
            self.server_thread.join(timeout=join_timeout)
            self.server_thread = None

    def get_current_frame(self, copy: bool = True) -> SensorFrame | None:
        with self.state.lock:
            if self.state.latest_frame is None:
                return None

            return copy_sensor_frame(self.state.latest_frame) if copy else self.state.latest_frame

    def get_current_depth_image(self, copy: bool = True) -> np.ndarray | None:
        frame = self.get_current_frame(copy=copy)
        return None if frame is None else frame.depth

    def get_current_infrared_image(self, copy: bool = True) -> np.ndarray | None:
        frame = self.get_current_frame(copy=copy)
        return None if frame is None else frame.infrared

    def get_current_images(self, copy: bool = True) -> tuple[np.ndarray, np.ndarray] | None:
        frame = self.get_current_frame(copy=copy)
        if frame is None:
            return None

        return frame.depth, frame.infrared

    def wait_for_frame(self, timeout: float | None = None, copy: bool = True) -> SensorFrame | None:
        with self.state.lock:
            if self.state.latest_frame is None:
                self.state.lock.wait(timeout=timeout)

            if self.state.latest_frame is None:
                return None

            return copy_sensor_frame(self.state.latest_frame) if copy else self.state.latest_frame

    def wait_for_next_frame(self, timeout: float | None = None, copy: bool = True) -> SensorFrame | None:
        with self.state.lock:
            previous_sequence = None if self.state.latest_frame is None else self.state.latest_frame.sequence
            deadline = None if timeout is None else time.monotonic() + timeout

            while True:
                current_frame = self.state.latest_frame
                if current_frame is not None and current_frame.sequence != previous_sequence:
                    return copy_sensor_frame(current_frame) if copy else current_frame

                if deadline is None:
                    self.state.lock.wait()
                    continue

                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None

                self.state.lock.wait(timeout=remaining)

    def wait_for_images(self, timeout: float | None = None, copy: bool = True) -> tuple[np.ndarray, np.ndarray] | None:
        frame = self.wait_for_frame(timeout=timeout, copy=copy)
        if frame is None:
            return None

        return frame.depth, frame.infrared

    def wait_for_next_images(self, timeout: float | None = None, copy: bool = True) -> tuple[np.ndarray, np.ndarray] | None:
        frame = self.wait_for_next_frame(timeout=timeout, copy=copy)
        if frame is None:
            return None

        return frame.depth, frame.infrared

    def get_receive_fps(self) -> float:
        with self.state.lock:
            return self.state.receive_fps

    def get_last_error(self) -> str:
        with self.state.lock:
            return self.state.last_error

    def _raw_sensor_worker(self, message: bytes) -> bytes:
        if message == b"quit":
            self.stop_event.set()
            return b"ok"

        try:
            if self.print_frame_log:
                return process_message(message, self.state, self.fps_counter)

            depth, infrared, sequence, client_timestamp = parse_sensor_images(message)
            received_timestamp = time.time()
            receive_fps = self.fps_counter.tick(time.perf_counter())
            frame = SensorFrame(depth, infrared, sequence, client_timestamp, received_timestamp)
            with self.state.lock:
                self.state.latest_frame = frame
                self.state.receive_fps = receive_fps
                self.state.last_error = ""
                self.state.lock.notify_all()

            return b"ok"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            with self.state.lock:
                self.state.last_error = error
                self.state.lock.notify_all()
            print(f"Packet error: {error}", flush=True)
            return b"error"

    def _server_loop(self) -> None:
        print(f"Raw sensor server listening on {self.host}:{self.port}", flush=True)
        self.server = RawSensorTcpServer(
            self.host,
            self.port,
            self._raw_sensor_worker,
            quit_token=b"quit",
            client_timeout=self.client_timeout,
        )
        self.server.set_debug_mode(False)

        try:
            while not self.stop_event.is_set() and self.server.running:
                self.server._try_accept()
                self.server._acquire_all()
                self.server._response_all()
                self.server._kick_timeout()
                time.sleep(0.001)
        finally:
            if self.server is not None:
                self.server.running = False
                self.server._kick_all()
                self.server.server_socket.close()
                self.server = None


def server_loop(state: SharedState, stop_event: threading.Event) -> None:
    receiver = RawSensorImageReceiver()
    receiver.state = state
    receiver.stop_event = stop_event
    receiver._server_loop()


def visualization_loop(state: SharedState, stop_event: threading.Event) -> None:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WIDTH * 2, HEIGHT)
    next_render_time = 0.0

    while not stop_event.is_set():
        now = time.perf_counter()
        if now < next_render_time:
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                stop_event.set()
                break
            time.sleep(min(0.002, next_render_time - now))
            continue

        next_render_time = now + VISUALIZATION_INTERVAL_SECONDS

        with state.lock:
            frame = state.latest_frame
            receive_fps = state.receive_fps
            last_error = state.last_error

        if frame is None:
            display = render_waiting_frame(last_error or f"Listening on {HOST}:{PORT}")
        else:
            display = render_frame(frame, receive_fps)
            if last_error:
                draw_text(display, [last_error])

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            stop_event.set()
            break

    cv2.destroyAllWindows()


_default_receiver: RawSensorImageReceiver | None = None


def start_receiver(
    host: str = HOST,
    port: int = PORT,
    show_visualization: bool = False,
    print_frame_log: bool = True,
) -> RawSensorImageReceiver:
    global _default_receiver

    receiver = RawSensorImageReceiver(host=host, port=port, print_frame_log=print_frame_log)
    receiver.start()
    _default_receiver = receiver

    if show_visualization:
        visualization_loop(receiver.state, receiver.stop_event)

    return receiver


def stop_receiver() -> None:
    global _default_receiver

    if _default_receiver is not None:
        _default_receiver.stop()
        _default_receiver = None


def get_current_depth_image(copy: bool = True) -> np.ndarray | None:
    if _default_receiver is None:
        return None

    return _default_receiver.get_current_depth_image(copy=copy)


def get_current_infrared_image(copy: bool = True) -> np.ndarray | None:
    if _default_receiver is None:
        return None

    return _default_receiver.get_current_infrared_image(copy=copy)


def get_current_images(copy: bool = True) -> tuple[np.ndarray, np.ndarray] | None:
    if _default_receiver is None:
        return None

    return _default_receiver.get_current_images(copy=copy)


def wait_for_images(timeout: float | None = None, copy: bool = True) -> tuple[np.ndarray, np.ndarray] | None:
    if _default_receiver is None:
        return None

    return _default_receiver.wait_for_images(timeout=timeout, copy=copy)


def wait_for_next_images(timeout: float | None = None, copy: bool = True) -> tuple[np.ndarray, np.ndarray] | None:
    if _default_receiver is None:
        return None

    return _default_receiver.wait_for_next_images(timeout=timeout, copy=copy)


def main() -> None:
    receiver = RawSensorImageReceiver()

    def request_shutdown(signum: int, frame: object) -> None:
        print(SHUTDOWN_MESSAGE, flush=True)
        receiver.stop_event.set()

    signal.signal(signal.SIGINT, request_shutdown)
    receiver.start()

    try:
        print("Press Ctrl+C in this terminal, or press Q/Esc in the image window, to exit.", flush=True)
        visualization_loop(receiver.state, receiver.stop_event)
    except KeyboardInterrupt:
        print(SHUTDOWN_MESSAGE, flush=True)
    finally:
        receiver.stop()


if __name__ == "__main__":
    main()
