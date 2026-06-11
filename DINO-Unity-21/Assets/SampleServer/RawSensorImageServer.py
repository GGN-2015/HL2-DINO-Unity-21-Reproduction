from __future__ import annotations

print("Server initializing ...", flush=True)

import argparse
import json
import pickle
import queue
import signal
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


DEFAULT_HOST = "169.254.83.86"
DEFAULT_PORT = 8888
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_DETECTOR = "threshold"
DEFAULT_THRESHOLD_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_THRESHOLD_PERCENTILE = 99.7
DEFAULT_THRESHOLD_MIN_THRESHOLD = 0
DEFAULT_THRESHOLD_MIN_AREA = 6
DEFAULT_THRESHOLD_MAX_AREA = 3000
DEFAULT_THRESHOLD_MIN_CIRCULARITY = 0.15
DEFAULT_THRESHOLD_MIN_ASPECT_RATIO = 0.25
DEFAULT_THRESHOLD_MAX_ASPECT_RATIO = 4.0
DEFAULT_THRESHOLD_MIN_WIDTH = 1
DEFAULT_THRESHOLD_MIN_HEIGHT = 1
DEFAULT_THRESHOLD_MORPHOLOGY_KERNEL_SIZE = 0
DEFAULT_THRESHOLD_MORPHOLOGY_OPEN_ITERATIONS = 0
DEFAULT_THRESHOLD_MAX_MARKERS = 16
SERVER_CONFIG_PATH = Path(__file__).with_name("RawSensorImageServerConfig.json")


def _normalize_detector_name(detector: str) -> str:
    normalized = detector.lower()
    if normalized == "blob":
        return "threshold"
    return normalized


def load_server_config(config_path: Path = SERVER_CONFIG_PATH) -> dict[str, str | int | float]:
    defaults = {
        "host": DEFAULT_HOST,
        "port": DEFAULT_PORT,
        "width": DEFAULT_WIDTH,
        "height": DEFAULT_HEIGHT,
        "detector": DEFAULT_DETECTOR,
        "threshold_confidence_threshold": DEFAULT_THRESHOLD_CONFIDENCE_THRESHOLD,
        "threshold_percentile": DEFAULT_THRESHOLD_PERCENTILE,
        "threshold_min_threshold": DEFAULT_THRESHOLD_MIN_THRESHOLD,
        "threshold_min_area": DEFAULT_THRESHOLD_MIN_AREA,
        "threshold_max_area": DEFAULT_THRESHOLD_MAX_AREA,
        "threshold_min_circularity": DEFAULT_THRESHOLD_MIN_CIRCULARITY,
        "threshold_min_aspect_ratio": DEFAULT_THRESHOLD_MIN_ASPECT_RATIO,
        "threshold_max_aspect_ratio": DEFAULT_THRESHOLD_MAX_ASPECT_RATIO,
        "threshold_min_width": DEFAULT_THRESHOLD_MIN_WIDTH,
        "threshold_min_height": DEFAULT_THRESHOLD_MIN_HEIGHT,
        "threshold_morphology_kernel_size": DEFAULT_THRESHOLD_MORPHOLOGY_KERNEL_SIZE,
        "threshold_morphology_open_iterations": DEFAULT_THRESHOLD_MORPHOLOGY_OPEN_ITERATIONS,
        "threshold_max_markers": DEFAULT_THRESHOLD_MAX_MARKERS,
    }
    if not config_path.exists():
        return defaults

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Server config must be a JSON object: {config_path}")

    loaded_config = {
        "host": str(config.get("host", defaults["host"])),
        "port": int(config.get("port", defaults["port"])),
        "width": int(config.get("width", defaults["width"])),
        "height": int(config.get("height", defaults["height"])),
        "detector": _normalize_detector_name(str(config.get("detector", defaults["detector"]))),
        "threshold_confidence_threshold": float(
            config.get("threshold_confidence_threshold", defaults["threshold_confidence_threshold"])
        ),
        "threshold_percentile": float(
            config.get(
                "threshold_percentile",
                config.get("blob_threshold_percentile", defaults["threshold_percentile"]),
            )
        ),
        "threshold_min_threshold": int(
            config.get("threshold_min_threshold", config.get("blob_min_threshold", defaults["threshold_min_threshold"]))
        ),
        "threshold_min_area": int(
            config.get("threshold_min_area", config.get("blob_min_area", defaults["threshold_min_area"]))
        ),
        "threshold_max_area": int(
            config.get("threshold_max_area", config.get("blob_max_area", defaults["threshold_max_area"]))
        ),
        "threshold_min_circularity": float(
            config.get("threshold_min_circularity", defaults["threshold_min_circularity"])
        ),
        "threshold_min_aspect_ratio": float(
            config.get("threshold_min_aspect_ratio", defaults["threshold_min_aspect_ratio"])
        ),
        "threshold_max_aspect_ratio": float(
            config.get("threshold_max_aspect_ratio", defaults["threshold_max_aspect_ratio"])
        ),
        "threshold_min_width": int(config.get("threshold_min_width", defaults["threshold_min_width"])),
        "threshold_min_height": int(config.get("threshold_min_height", defaults["threshold_min_height"])),
        "threshold_morphology_kernel_size": int(
            config.get("threshold_morphology_kernel_size", defaults["threshold_morphology_kernel_size"])
        ),
        "threshold_morphology_open_iterations": int(
            config.get("threshold_morphology_open_iterations", defaults["threshold_morphology_open_iterations"])
        ),
        "threshold_max_markers": int(
            config.get("threshold_max_markers", config.get("blob_max_markers", defaults["threshold_max_markers"]))
        ),
    }
    if loaded_config["port"] <= 0:
        raise ValueError(f"Server config port must be positive: {loaded_config['port']}")
    if loaded_config["width"] <= 0 or loaded_config["height"] <= 0:
        raise ValueError(
            f"Server config image size must be positive: {loaded_config['width']}x{loaded_config['height']}"
        )
    if loaded_config["detector"] not in ("threshold", "yolo"):
        raise ValueError(f"Server config detector must be 'threshold' or 'yolo': {loaded_config['detector']}")
    if not 0.0 <= loaded_config["threshold_confidence_threshold"] <= 1.0:
        raise ValueError(
            "Server config threshold_confidence_threshold must be between 0 and 1: "
            f"{loaded_config['threshold_confidence_threshold']}"
        )
    if not 0.0 <= loaded_config["threshold_percentile"] <= 100.0:
        raise ValueError(
            "Server config threshold_percentile must be between 0 and 100: "
            f"{loaded_config['threshold_percentile']}"
        )
    if loaded_config["threshold_min_threshold"] < 0:
        raise ValueError(
            "Server config threshold_min_threshold must be non-negative: "
            f"{loaded_config['threshold_min_threshold']}"
        )
    if (
        loaded_config["threshold_min_area"] <= 0
        or loaded_config["threshold_max_area"] < loaded_config["threshold_min_area"]
    ):
        raise ValueError(
            "Server config threshold area bounds must satisfy 0 < min <= max: "
            f"{loaded_config['threshold_min_area']}..{loaded_config['threshold_max_area']}"
        )
    if loaded_config["threshold_min_circularity"] < 0.0:
        raise ValueError(
            "Server config threshold_min_circularity must be non-negative: "
            f"{loaded_config['threshold_min_circularity']}"
        )
    if (
        loaded_config["threshold_min_aspect_ratio"] <= 0.0
        or loaded_config["threshold_max_aspect_ratio"] < loaded_config["threshold_min_aspect_ratio"]
    ):
        raise ValueError(
            "Server config threshold aspect ratio bounds must satisfy 0 < min <= max: "
            f"{loaded_config['threshold_min_aspect_ratio']}..{loaded_config['threshold_max_aspect_ratio']}"
        )
    if loaded_config["threshold_min_width"] <= 0 or loaded_config["threshold_min_height"] <= 0:
        raise ValueError(
            "Server config threshold minimum dimensions must be positive: "
            f"{loaded_config['threshold_min_width']}x{loaded_config['threshold_min_height']}"
        )
    if loaded_config["threshold_morphology_kernel_size"] < 0:
        raise ValueError(
            "Server config threshold_morphology_kernel_size must be non-negative: "
            f"{loaded_config['threshold_morphology_kernel_size']}"
        )
    if loaded_config["threshold_morphology_open_iterations"] < 0:
        raise ValueError(
            "Server config threshold_morphology_open_iterations must be non-negative: "
            f"{loaded_config['threshold_morphology_open_iterations']}"
        )
    if loaded_config["threshold_max_markers"] < 0:
        raise ValueError(
            "Server config threshold_max_markers must be non-negative: "
            f"{loaded_config['threshold_max_markers']}"
        )

    return loaded_config


SERVER_CONFIG = load_server_config()
HOST = str(SERVER_CONFIG["host"])
PORT = int(SERVER_CONFIG["port"])
WIDTH = int(SERVER_CONFIG["width"])
HEIGHT = int(SERVER_CONFIG["height"])
PIXEL_COUNT = WIDTH * HEIGHT
DETECTOR = str(SERVER_CONFIG["detector"])
THRESHOLD_CONFIDENCE_THRESHOLD = float(SERVER_CONFIG["threshold_confidence_threshold"])
THRESHOLD_PERCENTILE = float(SERVER_CONFIG["threshold_percentile"])
THRESHOLD_MIN_THRESHOLD = int(SERVER_CONFIG["threshold_min_threshold"])
THRESHOLD_MIN_AREA = int(SERVER_CONFIG["threshold_min_area"])
THRESHOLD_MAX_AREA = int(SERVER_CONFIG["threshold_max_area"])
THRESHOLD_MIN_CIRCULARITY = float(SERVER_CONFIG["threshold_min_circularity"])
THRESHOLD_MIN_ASPECT_RATIO = float(SERVER_CONFIG["threshold_min_aspect_ratio"])
THRESHOLD_MAX_ASPECT_RATIO = float(SERVER_CONFIG["threshold_max_aspect_ratio"])
THRESHOLD_MIN_WIDTH = int(SERVER_CONFIG["threshold_min_width"])
THRESHOLD_MIN_HEIGHT = int(SERVER_CONFIG["threshold_min_height"])
THRESHOLD_MORPHOLOGY_KERNEL_SIZE = int(SERVER_CONFIG["threshold_morphology_kernel_size"])
THRESHOLD_MORPHOLOGY_OPEN_ITERATIONS = int(SERVER_CONFIG["threshold_morphology_open_iterations"])
THRESHOLD_MAX_MARKERS = int(SERVER_CONFIG["threshold_max_markers"])

import cv2
import numpy as np
from ir_yolo_tracker import IRMarkerTracker, MarkerDetection, ThresholdMarkerDetector, create_tracker, draw_detections
from simple_tcp_server import SimpleTcpServer

RAW_STREAM_PREFIX = b"raw_stream:"
MAGIC_V1 = b"DINOIMG1"
MAGIC_V2 = b"DINOIMG2"
MAGIC_V3 = b"DINOIMG3"
MAGIC_V4 = b"DINOIMG4"
HEADER_V1_STRUCT = struct.Struct("<8siiQdii")
HEADER_V2_STRUCT = struct.Struct("<8siiQdiii")
HEADER_V3_STRUCT = struct.Struct("<8siiQdiiii")
HEADER_V4_STRUCT = struct.Struct("<8siiQdiiii")
MARKER_PIXEL_RESPONSE_MAGIC = b"DINOUV01"
MARKER_PIXEL_RESPONSE_HEADER_STRUCT = struct.Struct("<8sI")
MARKER_PIXEL_RESPONSE_POINT_STRUCT = struct.Struct("<ff")
DEPTH_TO_WORLD_MATRIX_VALUES = 16
DEPTH_MIN_MM = 1
DEPTH_MAX_MM = 4090
WINDOW_NAME = "DINO Raw Sensor Stream"
SHUTDOWN_MESSAGE = "Shutdown requested. Closing raw sensor server..."
VISUALIZATION_INTERVAL_SECONDS = 1.0 / 30.0
EXIT_HINT_TEXT = "press Q/Esc in the image window, to exit"
PROCESSING_AVERAGE_SAMPLE_COUNT = 120
PROJECT_ROOT = Path(__file__).resolve().parents[3]
IR_DATA_DIR = PROJECT_ROOT / "IRData"
SAVE_RAW_IMAGES_TO_PICKLE = False
PRINT_FRAME_LOG = False
DEPTH_SAVE_DIR_NAME = "depth"
INFRARED_SAVE_DIR_NAME = "infrared"


@dataclass
class SensorFrame:
    depth: np.ndarray
    infrared: np.ndarray
    sequence: int
    client_timestamp: float
    received_timestamp: float
    depth_to_world_matrix: np.ndarray | None = None
    unit_plane_map: np.ndarray | None = None
    marker_detections: list[MarkerDetection] = field(default_factory=list)


@dataclass
class SharedState:
    lock: threading.Condition = field(default_factory=threading.Condition)
    active_tcp_connections: int = 0
    latest_frame: SensorFrame | None = None
    receive_fps: float = 0.0
    average_yolo_marker_ms: float = 0.0
    last_response_latency_ms: float = 0.0
    average_response_latency_ms: float = 0.0
    last_error: str = ""


class RawSensorTcpServer(SimpleTcpServer):
    def __init__(
        self,
        *args,
        state: SharedState | None = None,
        on_disconnect=None,
        print_response_latency: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.state = state
        self.on_disconnect = on_disconnect
        self.print_response_latency = print_response_latency
        self.response_latency_samples: deque[float] = deque(maxlen=PROCESSING_AVERAGE_SAMPLE_COUNT)

    def _handle(self, conn, addr) -> None:
        connected = datetime.now()
        print(f"[{connected:%Y-%m-%d %H:%M:%S.%f}] Client connected from {addr[0]}:{addr[1]}", flush=True)
        if self.state is not None:
            with self.state.lock:
                self.state.active_tcp_connections += 1
                self.state.last_error = ""
                self.state.lock.notify_all()
        super()._handle(conn, addr)

    def _conn_close(self, addr) -> None:
        print(f"Client disconnected from {addr[0]}:{addr[1]}", flush=True)
        if self.state is not None:
            clear_connection_frame(self.state)
        if self.on_disconnect is not None:
            self.on_disconnect()
        super()._conn_close(addr)

    def _calc_and_resp(self, addr, msg_now: bytes) -> bool:
        response_start = time.perf_counter()
        died = super()._calc_and_resp(addr, msg_now)
        if died or msg_now == self.quit_token:
            return died

        latency_ms = (time.perf_counter() - response_start) * 1000.0
        self.response_latency_samples.append(latency_ms)
        average_latency_ms = sum(self.response_latency_samples) / len(self.response_latency_samples)
        if self.state is not None:
            with self.state.lock:
                self.state.last_response_latency_ms = latency_ms
                self.state.average_response_latency_ms = average_latency_ms
                self.state.lock.notify_all()

        if self.print_response_latency:
            print(
                f"response_sent addr={addr[0]}:{addr[1]} "
                f"server_latency_ms={latency_ms:.2f} avg_server_latency_ms={average_latency_ms:.2f}",
                flush=True,
            )

        return died


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


class RollingAverage:
    def __init__(self, max_samples: int = PROCESSING_AVERAGE_SAMPLE_COUNT) -> None:
        self.max_samples = max_samples
        self.samples: deque[float] = deque(maxlen=max_samples)
        self.lock = threading.Lock()

    def add(self, value: float) -> float:
        with self.lock:
            self.samples.append(value)
            return sum(self.samples) / len(self.samples)

    def reset(self) -> None:
        with self.lock:
            self.samples.clear()


def create_marker_tracker(detector: str) -> IRMarkerTracker | ThresholdMarkerDetector:
    detector = _normalize_detector_name(detector)
    if detector == "yolo":
        return create_tracker()
    if detector == "threshold":
        max_detections = None if THRESHOLD_MAX_MARKERS <= 0 else THRESHOLD_MAX_MARKERS
        return ThresholdMarkerDetector(
            confidence_threshold=THRESHOLD_CONFIDENCE_THRESHOLD,
            threshold_percentile=THRESHOLD_PERCENTILE,
            minimum_threshold=THRESHOLD_MIN_THRESHOLD,
            min_area=THRESHOLD_MIN_AREA,
            max_area=THRESHOLD_MAX_AREA,
            min_circularity=THRESHOLD_MIN_CIRCULARITY,
            min_aspect_ratio=THRESHOLD_MIN_ASPECT_RATIO,
            max_aspect_ratio=THRESHOLD_MAX_ASPECT_RATIO,
            min_width=THRESHOLD_MIN_WIDTH,
            min_height=THRESHOLD_MIN_HEIGHT,
            morphology_kernel_size=THRESHOLD_MORPHOLOGY_KERNEL_SIZE,
            morphology_open_iterations=THRESHOLD_MORPHOLOGY_OPEN_ITERATIONS,
            max_detections=max_detections,
        )
    raise ValueError(f"Unsupported detector: {detector}")


@dataclass(frozen=True)
class RawImageSaveTask:
    path: Path
    image: np.ndarray


class AsyncRawImagePickleWriter:
    def __init__(self, output_dir: Path | str = IR_DATA_DIR) -> None:
        self.output_dir = Path(output_dir)
        self.depth_dir = self.output_dir / DEPTH_SAVE_DIR_NAME
        self.infrared_dir = self.output_dir / INFRARED_SAVE_DIR_NAME
        self.tasks: queue.Queue[RawImageSaveTask | None] = queue.Queue()
        self.counter_lock = threading.Lock()
        self.next_frame_number = 1
        self.thread: threading.Thread | None = None
        self.last_error = ""

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        self.infrared_dir.mkdir(parents=True, exist_ok=True)
        self.next_frame_number = self._discover_next_frame_number()
        self.tasks = queue.Queue()
        self.last_error = ""
        self.thread = threading.Thread(target=self._worker_loop, name="RawImagePickleWriter", daemon=True)
        self.thread.start()
        print(f"Raw image pickle saving enabled: {self.output_dir}", flush=True)

    def stop(self, join_timeout: float | None = None) -> None:
        if self.thread is None:
            return

        self.tasks.put_nowait(None)
        self.thread.join(timeout=join_timeout)
        if self.thread.is_alive():
            print("Raw image pickle writer is still flushing queued images.", flush=True)
            return

        self.thread = None

    def submit_frame(self, frame: SensorFrame) -> int | None:
        if self.thread is None or not self.thread.is_alive():
            return None

        with self.counter_lock:
            frame_number = self.next_frame_number
            self.next_frame_number += 1

        filename = f"{frame_number:07d}.pickle"
        self.tasks.put_nowait(RawImageSaveTask(self.depth_dir / filename, frame.depth))
        self.tasks.put_nowait(RawImageSaveTask(self.infrared_dir / filename, frame.infrared))
        return frame_number

    def _discover_next_frame_number(self) -> int:
        max_frame_number = 0
        for directory in (self.depth_dir, self.infrared_dir):
            if not directory.exists():
                continue

            for path in directory.glob("*.pickle"):
                if path.stem.isdecimal():
                    max_frame_number = max(max_frame_number, int(path.stem))

        return max_frame_number + 1

    def _worker_loop(self) -> None:
        while True:
            task = self.tasks.get()
            try:
                if task is None:
                    return

                image = np.asarray(task.image, dtype=np.uint16)
                if image.shape != (HEIGHT, WIDTH):
                    raise ValueError(f"Unexpected image shape for {task.path}: {image.shape}")

                with task.path.open("wb") as file:
                    pickle.dump(image, file, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                print(f"Raw image pickle save error: {self.last_error}", flush=True)
            finally:
                self.tasks.task_done()


def copy_sensor_frame(frame: SensorFrame) -> SensorFrame:
    return SensorFrame(
        depth=frame.depth.copy(),
        infrared=frame.infrared.copy(),
        sequence=frame.sequence,
        client_timestamp=frame.client_timestamp,
        received_timestamp=frame.received_timestamp,
        depth_to_world_matrix=None if frame.depth_to_world_matrix is None else frame.depth_to_world_matrix.copy(),
        unit_plane_map=None if frame.unit_plane_map is None else frame.unit_plane_map.copy(),
        marker_detections=list(frame.marker_detections),
    )


def clear_connection_frame(state: SharedState) -> None:
    with state.lock:
        state.active_tcp_connections = max(0, state.active_tcp_connections - 1)
        state.latest_frame = None
        state.receive_fps = 0.0
        state.average_yolo_marker_ms = 0.0
        state.last_response_latency_ms = 0.0
        state.average_response_latency_ms = 0.0
        state.last_error = ""
        state.lock.notify_all()


def parse_sensor_images(
    message: bytes | memoryview,
    cached_unit_plane_map: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int, float, np.ndarray | None, np.ndarray | None]:
    if len(message) < HEADER_V1_STRUCT.size:
        raise ValueError(f"Message is too short: {len(message)} bytes")

    magic = bytes(message[:8])
    if magic == MAGIC_V1:
        return parse_sensor_images_v1(message)
    if magic == MAGIC_V2:
        return parse_sensor_images_v2(message)
    if magic == MAGIC_V3:
        return parse_sensor_images_v3(message)
    if magic == MAGIC_V4:
        return parse_sensor_images_v4(message, cached_unit_plane_map)

    raise ValueError(f"Bad packet magic: {magic!r}")


def parse_sensor_images_v1(
    message: bytes | memoryview,
) -> tuple[np.ndarray, np.ndarray, int, float, np.ndarray | None, np.ndarray | None]:
    magic, width, height, sequence, client_timestamp, depth_count, infrared_count = HEADER_V1_STRUCT.unpack_from(message, 0)
    if magic != MAGIC_V1:
        raise ValueError(f"Bad packet magic: {magic!r}")
    if width != WIDTH or height != HEIGHT:
        raise ValueError(f"Unexpected image size: {width}x{height}")
    if depth_count != PIXEL_COUNT or infrared_count != PIXEL_COUNT:
        raise ValueError(f"Unexpected pixel counts: depth={depth_count}, infrared={infrared_count}")

    depth_bytes = depth_count * np.dtype(np.uint16).itemsize
    infrared_bytes = infrared_count * np.dtype(np.uint16).itemsize
    expected_size = HEADER_V1_STRUCT.size + depth_bytes + infrared_bytes
    if len(message) != expected_size:
        raise ValueError(f"Unexpected packet size: got {len(message)}, expected {expected_size}")

    offset = HEADER_V1_STRUCT.size
    depth = np.frombuffer(message, dtype="<u2", count=depth_count, offset=offset).reshape((height, width))
    offset += depth_bytes
    infrared = np.frombuffer(message, dtype="<u2", count=infrared_count, offset=offset).reshape((height, width))

    return depth, infrared, sequence, client_timestamp, None, None


def parse_sensor_images_v2(message: bytes | memoryview) -> tuple[np.ndarray, np.ndarray, int, float, np.ndarray, np.ndarray | None]:
    (
        magic,
        width,
        height,
        sequence,
        client_timestamp,
        depth_count,
        infrared_count,
        matrix_count,
    ) = HEADER_V2_STRUCT.unpack_from(message, 0)
    if magic != MAGIC_V2:
        raise ValueError(f"Bad packet magic: {magic!r}")
    if width != WIDTH or height != HEIGHT:
        raise ValueError(f"Unexpected image size: {width}x{height}")
    if depth_count != PIXEL_COUNT or infrared_count != PIXEL_COUNT:
        raise ValueError(f"Unexpected pixel counts: depth={depth_count}, infrared={infrared_count}")
    if matrix_count != DEPTH_TO_WORLD_MATRIX_VALUES:
        raise ValueError(f"Unexpected depth-to-world matrix value count: {matrix_count}")

    matrix_bytes = matrix_count * np.dtype(np.float64).itemsize
    depth_bytes = depth_count * np.dtype(np.uint16).itemsize
    infrared_bytes = infrared_count * np.dtype(np.uint16).itemsize
    expected_size = HEADER_V2_STRUCT.size + matrix_bytes + depth_bytes + infrared_bytes
    if len(message) != expected_size:
        raise ValueError(f"Unexpected packet size: got {len(message)}, expected {expected_size}")

    offset = HEADER_V2_STRUCT.size
    depth_to_world_matrix = np.frombuffer(message, dtype="<f8", count=matrix_count, offset=offset).reshape(
        (4, 4), order="F"
    )
    offset += matrix_bytes
    depth = np.frombuffer(message, dtype="<u2", count=depth_count, offset=offset).reshape((height, width))
    offset += depth_bytes
    infrared = np.frombuffer(message, dtype="<u2", count=infrared_count, offset=offset).reshape((height, width))

    return depth, infrared, sequence, client_timestamp, depth_to_world_matrix, None


def parse_sensor_images_v3(message: bytes | memoryview) -> tuple[np.ndarray, np.ndarray, int, float, np.ndarray, np.ndarray]:
    (
        magic,
        width,
        height,
        sequence,
        client_timestamp,
        depth_count,
        infrared_count,
        matrix_count,
        unit_plane_count,
    ) = HEADER_V3_STRUCT.unpack_from(message, 0)
    if magic != MAGIC_V3:
        raise ValueError(f"Bad packet magic: {magic!r}")
    if width != WIDTH or height != HEIGHT:
        raise ValueError(f"Unexpected image size: {width}x{height}")
    if depth_count != PIXEL_COUNT or infrared_count != PIXEL_COUNT:
        raise ValueError(f"Unexpected pixel counts: depth={depth_count}, infrared={infrared_count}")
    if matrix_count != DEPTH_TO_WORLD_MATRIX_VALUES:
        raise ValueError(f"Unexpected depth-to-world matrix value count: {matrix_count}")
    if unit_plane_count != PIXEL_COUNT:
        raise ValueError(f"Unexpected unit-plane pixel count: {unit_plane_count}")

    matrix_bytes = matrix_count * np.dtype(np.float64).itemsize
    unit_plane_bytes = unit_plane_count * 2 * np.dtype(np.float32).itemsize
    depth_bytes = depth_count * np.dtype(np.uint16).itemsize
    infrared_bytes = infrared_count * np.dtype(np.uint16).itemsize
    expected_size = HEADER_V3_STRUCT.size + matrix_bytes + unit_plane_bytes + depth_bytes + infrared_bytes
    if len(message) != expected_size:
        raise ValueError(f"Unexpected packet size: got {len(message)}, expected {expected_size}")

    offset = HEADER_V3_STRUCT.size
    depth_to_world_matrix = np.frombuffer(message, dtype="<f8", count=matrix_count, offset=offset).reshape(
        (4, 4), order="F"
    )
    offset += matrix_bytes
    unit_plane_map = np.frombuffer(message, dtype="<f4", count=unit_plane_count * 2, offset=offset).reshape(
        (height, width, 2)
    )
    offset += unit_plane_bytes
    depth = np.frombuffer(message, dtype="<u2", count=depth_count, offset=offset).reshape((height, width))
    offset += depth_bytes
    infrared = np.frombuffer(message, dtype="<u2", count=infrared_count, offset=offset).reshape((height, width))

    return depth, infrared, sequence, client_timestamp, depth_to_world_matrix, unit_plane_map


def parse_sensor_images_v4(
    message: bytes | memoryview,
    cached_unit_plane_map: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, int, float, np.ndarray, np.ndarray | None]:
    (
        magic,
        width,
        height,
        sequence,
        client_timestamp,
        depth_count,
        infrared_count,
        matrix_count,
        unit_plane_count,
    ) = HEADER_V4_STRUCT.unpack_from(message, 0)
    if magic != MAGIC_V4:
        raise ValueError(f"Bad packet magic: {magic!r}")
    if width != WIDTH or height != HEIGHT:
        raise ValueError(f"Unexpected image size: {width}x{height}")
    if depth_count != PIXEL_COUNT or infrared_count != PIXEL_COUNT:
        raise ValueError(f"Unexpected pixel counts: depth={depth_count}, infrared={infrared_count}")
    if matrix_count != DEPTH_TO_WORLD_MATRIX_VALUES:
        raise ValueError(f"Unexpected depth-to-world matrix value count: {matrix_count}")
    if unit_plane_count not in (0, PIXEL_COUNT):
        raise ValueError(f"Unexpected unit-plane pixel count: {unit_plane_count}")

    matrix_bytes = matrix_count * np.dtype(np.float64).itemsize
    unit_plane_bytes = unit_plane_count * 2 * np.dtype(np.float32).itemsize
    depth_bytes = depth_count * np.dtype(np.uint16).itemsize
    infrared_bytes = infrared_count * np.dtype(np.uint16).itemsize
    expected_size = HEADER_V4_STRUCT.size + matrix_bytes + unit_plane_bytes + depth_bytes + infrared_bytes
    if len(message) != expected_size:
        raise ValueError(f"Unexpected packet size: got {len(message)}, expected {expected_size}")

    offset = HEADER_V4_STRUCT.size
    depth_to_world_matrix = np.frombuffer(message, dtype="<f8", count=matrix_count, offset=offset).reshape(
        (4, 4), order="F"
    )
    offset += matrix_bytes
    if unit_plane_count:
        unit_plane_map = np.frombuffer(message, dtype="<f4", count=unit_plane_count * 2, offset=offset).reshape(
            (height, width, 2)
        )
    else:
        unit_plane_map = cached_unit_plane_map
    offset += unit_plane_bytes
    depth = np.frombuffer(message, dtype="<u2", count=depth_count, offset=offset).reshape((height, width))
    offset += depth_bytes
    infrared = np.frombuffer(message, dtype="<u2", count=infrared_count, offset=offset).reshape((height, width))

    return depth, infrared, sequence, client_timestamp, depth_to_world_matrix, unit_plane_map


def unwrap_raw_stream_request(message: bytes) -> memoryview:
    if not message.startswith(RAW_STREAM_PREFIX):
        raise ValueError("Unsupported request prefix")

    return memoryview(message)[len(RAW_STREAM_PREFIX):]


def packet_contains_unit_plane_map(message: bytes | memoryview) -> bool:
    if len(message) < HEADER_V1_STRUCT.size:
        return False

    magic = bytes(message[:8])
    if magic == MAGIC_V3:
        return True
    if magic != MAGIC_V4 or len(message) < HEADER_V4_STRUCT.size:
        return False

    unit_plane_count = HEADER_V4_STRUCT.unpack_from(message, 0)[8]
    return unit_plane_count == PIXEL_COUNT


def detect_infrared_markers(
    infrared: np.ndarray,
    marker_tracker,
) -> tuple[list[MarkerDetection], float]:
    start_time = time.perf_counter()
    detections = marker_tracker.detect(infrared)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return detections, elapsed_ms


def serialize_marker_pixels(marker_detections: list[MarkerDetection]) -> bytes:
    response = bytearray(
        MARKER_PIXEL_RESPONSE_HEADER_STRUCT.size
        + len(marker_detections) * MARKER_PIXEL_RESPONSE_POINT_STRUCT.size
    )
    MARKER_PIXEL_RESPONSE_HEADER_STRUCT.pack_into(response, 0, MARKER_PIXEL_RESPONSE_MAGIC, len(marker_detections))
    offset = MARKER_PIXEL_RESPONSE_HEADER_STRUCT.size
    for marker_detection in marker_detections:
        pixel_x, pixel_y = marker_detection.center_xy
        MARKER_PIXEL_RESPONSE_POINT_STRUCT.pack_into(
            response,
            offset,
            float(pixel_x),
            float(pixel_y),
        )
        offset += MARKER_PIXEL_RESPONSE_POINT_STRUCT.size
    return bytes(response)


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


def draw_exit_hint(image: np.ndarray) -> None:
    position = (10, image.shape[0] - 14)
    cv2.putText(image, EXIT_HINT_TEXT, position, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, EXIT_HINT_TEXT, position, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def render_frame(
    frame: SensorFrame,
    receive_fps: float,
    average_yolo_marker_ms: float,
    average_response_latency_ms: float,
) -> np.ndarray:
    depth_display = cv2.cvtColor(normalize_depth_for_display(frame.depth), cv2.COLOR_GRAY2BGR)
    infrared_display = cv2.cvtColor(normalize_min_max_for_display(frame.infrared), cv2.COLOR_GRAY2BGR)
    infrared_display = draw_detections(infrared_display, frame.marker_detections)

    received_text = datetime.fromtimestamp(frame.received_timestamp).strftime("%H:%M:%S.%f")[:-3]
    draw_text(
        depth_display,
        [
            "Depth 16-bit",
            f"Frame {frame.sequence}",
            f"RX FPS {receive_fps:.1f}",
            f"Detect {average_yolo_marker_ms:.1f} ms",
            f"Server latency {average_response_latency_ms:.1f} ms",
        ],
    )
    draw_text(
        infrared_display,
        [
            "Infrared 16-bit",
            f"Markers {len(frame.marker_detections)}",
            f"Detect {average_yolo_marker_ms:.1f} ms",
            f"Server latency {average_response_latency_ms:.1f} ms",
            f"Received {received_text}",
            f"Client TS {frame.client_timestamp:.3f}",
        ],
    )

    display = np.hstack((depth_display, infrared_display))
    draw_exit_hint(display)
    return display


def render_waiting_frame(title: str, message: str) -> np.ndarray:
    image = np.zeros((HEIGHT, WIDTH * 2, 3), dtype=np.uint8)
    draw_text(image, [title, message])
    draw_exit_hint(image)
    return image


def publish_sensor_frame(
    depth: np.ndarray,
    infrared: np.ndarray,
    sequence: int,
    client_timestamp: float,
    depth_to_world_matrix: np.ndarray | None,
    unit_plane_map: np.ndarray | None,
    state: SharedState,
    fps_counter: FpsCounter,
    image_writer: AsyncRawImagePickleWriter | None = None,
    marker_detections: list[MarkerDetection] | None = None,
    average_yolo_marker_ms: float | None = None,
) -> tuple[SensorFrame, float]:
    received_timestamp = time.time()
    receive_fps = fps_counter.tick(time.perf_counter())
    frame = SensorFrame(
        depth,
        infrared,
        sequence,
        client_timestamp,
        received_timestamp,
        depth_to_world_matrix=depth_to_world_matrix,
        unit_plane_map=unit_plane_map,
        marker_detections=marker_detections or [],
    )

    with state.lock:
        state.latest_frame = frame
        state.receive_fps = receive_fps
        if average_yolo_marker_ms is not None:
            state.average_yolo_marker_ms = average_yolo_marker_ms
        state.last_error = ""
        state.lock.notify_all()

    if image_writer is not None:
        image_writer.submit_frame(frame)

    return frame, receive_fps


def process_message(
    message: bytes,
    state: SharedState,
    fps_counter: FpsCounter,
    marker_tracker: IRMarkerTracker | ThresholdMarkerDetector,
    yolo_average: RollingAverage,
    image_writer: AsyncRawImagePickleWriter | None = None,
    cached_unit_plane_map: np.ndarray | None = None,
) -> bytes:
    if message == b"quit":
        return b"quit"

    raw_stream_message = unwrap_raw_stream_request(message)
    depth, infrared, sequence, client_timestamp, depth_to_world_matrix, unit_plane_map = parse_sensor_images(
        raw_stream_message,
        cached_unit_plane_map=cached_unit_plane_map,
    )
    marker_detections, yolo_marker_ms = detect_infrared_markers(infrared, marker_tracker)
    average_yolo_marker_ms = yolo_average.add(yolo_marker_ms)
    frame, receive_fps = publish_sensor_frame(
        depth,
        infrared,
        sequence,
        client_timestamp,
        depth_to_world_matrix,
        unit_plane_map,
        state,
        fps_counter,
        image_writer=image_writer,
        marker_detections=marker_detections,
        average_yolo_marker_ms=average_yolo_marker_ms,
    )
    received = datetime.fromtimestamp(frame.received_timestamp)
    with state.lock:
        last_response_latency_ms = state.last_response_latency_ms
        average_response_latency_ms = state.average_response_latency_ms
    print(
        f"[{received:%Y-%m-%d %H:%M:%S.%f}] "
        f"frame={frame.sequence} client_ts={frame.client_timestamp:.6f} "
        f"rx_fps={receive_fps:.2f} markers={len(frame.marker_detections)} "
        f"avg_detect_ms={average_yolo_marker_ms:.2f} "
        f"last_server_latency_ms={last_response_latency_ms:.2f} "
        f"avg_server_latency_ms={average_response_latency_ms:.2f} "
        f"depth_shape={frame.depth.shape} infrared_shape={frame.infrared.shape}",
        flush=True,
    )

    return serialize_marker_pixels(marker_detections)


class RawSensorImageReceiver:
    def __init__(
        self,
        host: str = HOST,
        port: int = PORT,
        client_timeout: float = 3600.0,
        print_frame_log: bool = PRINT_FRAME_LOG,
        save_raw_images: bool = SAVE_RAW_IMAGES_TO_PICKLE,
        image_output_dir: Path | str = IR_DATA_DIR,
        detector: str = DETECTOR,
        print_response_latency: bool = False,
    ) -> None:
        self.host = host
        self.port = port
        self.client_timeout = client_timeout
        self.print_frame_log = print_frame_log
        self.save_raw_images = save_raw_images
        self.image_output_dir = Path(image_output_dir)
        self.detector = detector
        self.print_response_latency = print_response_latency
        self.state = SharedState()
        self.stop_event = threading.Event()
        self.server_thread: threading.Thread | None = None
        self.server: RawSensorTcpServer | None = None
        self.fps_counter = FpsCounter()
        self.yolo_average = RollingAverage()
        self.cached_unit_plane_map: np.ndarray | None = None
        self.image_writer = AsyncRawImagePickleWriter(self.image_output_dir) if save_raw_images else None
        self.marker_tracker = create_marker_tracker(detector)

    def start(self) -> None:
        if self.server_thread is not None and self.server_thread.is_alive():
            return

        self.stop_event.clear()
        if self.image_writer is not None:
            self.image_writer.start()
        self.server_thread = threading.Thread(target=self._server_loop, name="RawSensorTcpServer", daemon=True)
        self.server_thread.start()

    def stop(self, join_timeout: float = 1.0) -> None:
        self.stop_event.set()
        if self.server is not None:
            self.server.running = False

        if self.server_thread is not None:
            self.server_thread.join(timeout=join_timeout)
            self.server_thread = None

        if self.image_writer is not None:
            self.image_writer.stop()

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

    def _reset_processing_averages(self) -> None:
        self.yolo_average.reset()
        self.cached_unit_plane_map = None

    def _raw_sensor_worker(self, message: bytes) -> bytes:
        if message == b"quit":
            self.stop_event.set()
            return b"ok"

        try:
            if self.print_frame_log:
                raw_stream_message = unwrap_raw_stream_request(message)
                has_unit_plane_map = packet_contains_unit_plane_map(raw_stream_message)
                response = process_message(
                    message,
                    self.state,
                    self.fps_counter,
                    self.marker_tracker,
                    self.yolo_average,
                    image_writer=self.image_writer,
                    cached_unit_plane_map=self.cached_unit_plane_map,
                )
                if has_unit_plane_map:
                    with self.state.lock:
                        if self.state.latest_frame is not None:
                            unit_plane_map = self.state.latest_frame.unit_plane_map
                            self.cached_unit_plane_map = None if unit_plane_map is None else unit_plane_map.copy()
                return response

            raw_stream_message = unwrap_raw_stream_request(message)
            has_unit_plane_map = packet_contains_unit_plane_map(raw_stream_message)
            depth, infrared, sequence, client_timestamp, depth_to_world_matrix, unit_plane_map = parse_sensor_images(
                raw_stream_message,
                cached_unit_plane_map=self.cached_unit_plane_map,
            )
            if has_unit_plane_map:
                self.cached_unit_plane_map = unit_plane_map.copy()
            marker_detections, yolo_marker_ms = detect_infrared_markers(infrared, self.marker_tracker)
            average_yolo_marker_ms = self.yolo_average.add(yolo_marker_ms)
            publish_sensor_frame(
                depth,
                infrared,
                sequence,
                client_timestamp,
                depth_to_world_matrix,
                unit_plane_map,
                self.state,
                self.fps_counter,
                image_writer=self.image_writer,
                marker_detections=marker_detections,
                average_yolo_marker_ms=average_yolo_marker_ms,
            )

            return serialize_marker_pixels(marker_detections)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            with self.state.lock:
                self.state.last_error = error
                self.state.lock.notify_all()
            print(f"Packet error: {error}", flush=True)
            return b"error"

    def _server_loop(self) -> None:
        self.server = RawSensorTcpServer(
            self.host,
            self.port,
            self._raw_sensor_worker,
            quit_token=b"quit",
            client_timeout=self.client_timeout,
            state=self.state,
            on_disconnect=self._reset_processing_averages,
            print_response_latency=self.print_response_latency,
        )
        self.server.set_debug_mode(False)
        print(f"Raw sensor server listening on {self.host}:{self.port}", flush=True)

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
    receiver = RawSensorImageReceiver(detector=DETECTOR)
    receiver.state = state
    receiver.stop_event = stop_event
    if receiver.image_writer is not None:
        receiver.image_writer.start()

    try:
        receiver._server_loop()
    finally:
        if receiver.image_writer is not None:
            receiver.image_writer.stop()


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
            average_yolo_marker_ms = state.average_yolo_marker_ms
            average_response_latency_ms = state.average_response_latency_ms
            last_error = state.last_error
            active_tcp_connections = state.active_tcp_connections

        if frame is None:
            if last_error:
                waiting_title = "Waiting for raw sensor frames"
                waiting_message = last_error
            elif active_tcp_connections <= 0:
                waiting_title = "Waiting for TCP connection"
                waiting_message = f"Listening on {HOST}:{PORT}"
            else:
                waiting_title = "Waiting for raw sensor frames"
                waiting_message = "TCP connected"
            display = render_waiting_frame(waiting_title, waiting_message)
        else:
            display = render_frame(frame, receive_fps, average_yolo_marker_ms, average_response_latency_ms)
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
    print_frame_log: bool = PRINT_FRAME_LOG,
    save_raw_images: bool = SAVE_RAW_IMAGES_TO_PICKLE,
    image_output_dir: Path | str = IR_DATA_DIR,
    detector: str = DETECTOR,
    print_response_latency: bool = False,
) -> RawSensorImageReceiver:
    global _default_receiver

    receiver = RawSensorImageReceiver(
        host=host,
        port=port,
        print_frame_log=print_frame_log,
        save_raw_images=save_raw_images,
        image_output_dir=image_output_dir,
        detector=detector,
        print_response_latency=print_response_latency,
    )
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Receive HoloLens 2 raw depth and infrared images over TCP.")
    parser.add_argument("--host", default=HOST, help=f"TCP host/IP to bind. Default: {HOST}")
    parser.add_argument("--port", type=int, default=PORT, help=f"TCP port to bind. Default: {PORT}")
    parser.add_argument(
        "--save-raw-images",
        action="store_true",
        default=SAVE_RAW_IMAGES_TO_PICKLE,
        help=f"Save every depth and infrared frame as pickle files under {IR_DATA_DIR}.",
    )
    parser.add_argument(
        "--image-output-dir",
        default=str(IR_DATA_DIR),
        help=f"Directory for saved pickle files. Default: {IR_DATA_DIR}",
    )
    parser.add_argument(
        "--print-frame-log",
        action="store_true",
        default=PRINT_FRAME_LOG,
        help="Print one receive/debug log line for every raw sensor frame.",
    )
    parser.add_argument(
        "--print-response-latency",
        action="store_true",
        help=(
            "Print one latency line after each response is sent. "
            "Measures from complete request receipt to successful socket send."
        ),
    )
    parser.add_argument(
        "--detector",
        choices=("threshold", "blob", "yolo"),
        default=DETECTOR,
        help=f"Marker detector to use. 'blob' is accepted as an alias for 'threshold'. Default: {DETECTOR}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receiver = RawSensorImageReceiver(
        host=args.host,
        port=args.port,
        print_frame_log=args.print_frame_log,
        save_raw_images=args.save_raw_images,
        image_output_dir=args.image_output_dir,
        detector=args.detector,
        print_response_latency=args.print_response_latency,
    )

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
