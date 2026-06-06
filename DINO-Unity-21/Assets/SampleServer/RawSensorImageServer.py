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
    lock: threading.Lock = field(default_factory=threading.Lock)
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

    received = datetime.fromtimestamp(received_timestamp)
    print(
        f"[{received:%Y-%m-%d %H:%M:%S.%f}] "
        f"frame={sequence} client_ts={client_timestamp:.6f} "
        f"rx_fps={receive_fps:.2f} depth_shape={depth.shape} infrared_shape={infrared.shape}",
        flush=True,
    )

    return b"ok"


def server_loop(state: SharedState, stop_event: threading.Event) -> None:
    fps_counter = FpsCounter()

    def raw_sensor_worker(message: bytes) -> bytes:
        if message == b"quit":
            stop_event.set()
            return b"ok"

        try:
            return process_message(message, state, fps_counter)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            with state.lock:
                state.last_error = error
            print(f"Packet error: {error}", flush=True)
            return b"error"

    print(f"Raw sensor server listening on {HOST}:{PORT}", flush=True)
    server = RawSensorTcpServer(HOST, PORT, raw_sensor_worker, quit_token=b"quit", client_timeout=3600.0)
    server.set_debug_mode(False)

    while not stop_event.is_set() and server.running:
        server._try_accept()
        server._acquire_all()
        server._response_all()
        server._kick_timeout()
        time.sleep(0.001)

    server.running = False
    server._kick_all()
    server.server_socket.close()


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


def main() -> None:
    state = SharedState()
    stop_event = threading.Event()

    def request_shutdown(signum: int, frame: object) -> None:
        print(SHUTDOWN_MESSAGE, flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, request_shutdown)

    server_thread = threading.Thread(target=server_loop, args=(state, stop_event), name="RawSensorTcpServer", daemon=True)
    server_thread.start()

    try:
        print("Press Ctrl+C in this terminal, or press Q/Esc in the image window, to exit.", flush=True)
        visualization_loop(state, stop_event)
    except KeyboardInterrupt:
        print(SHUTDOWN_MESSAGE, flush=True)
    finally:
        stop_event.set()
        server_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
