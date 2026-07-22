from __future__ import annotations

import importlib
import logging
from pathlib import Path
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nero_collection.config import CameraConfig
from nero_collection.time_utils import now_us

log = logging.getLogger(__name__)


class CameraUnavailable(RuntimeError):
    pass


@dataclass
class CameraFrame:
    camera_name: str
    timestamp_us: int
    frame: np.ndarray


class CameraSource:
    name: str

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def poll(self) -> CameraFrame | None:
        raise NotImplementedError


@dataclass
class MockCamera(CameraSource):
    config: CameraConfig
    name: str = field(init=False)
    _next_frame_t: float = field(default=0.0)
    _counter: int = 0

    def __post_init__(self) -> None:
        self.name = self.config.name

    def start(self) -> None:
        self._next_frame_t = time.monotonic()

    def stop(self) -> None:
        return None

    def poll(self) -> CameraFrame | None:
        now = time.monotonic()
        if now < self._next_frame_t:
            return None
        period = 1.0 / max(self.config.fps, 1.0)
        self._next_frame_t = now + period
        width, height = self.config.output_size or (self.config.width, self.config.height)
        yy, xx = np.mgrid[0:height, 0:width]
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[..., 0] = (xx + self._counter) % 255
        frame[..., 1] = (yy + 2 * self._counter) % 255
        frame[..., 2] = (40 + 3 * self._counter) % 255
        self._counter += 1
        return CameraFrame(self.name, now_us(), frame)


@dataclass
class OrbbecDabaiCamera(CameraSource):
    config: CameraConfig
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = self.config.name
        module = _import_orbbec_module()
        if module is None:
            raise CameraUnavailable("Orbbec_DaBai_SDK is not installed")
        raise CameraUnavailable(
            "Orbbec_DaBai_SDK is installed, but this project needs the local camera API binding added in OrbbecDabaiCamera"
        )

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def poll(self) -> CameraFrame | None:
        return None


@dataclass
class V4L2Camera(CameraSource):
    config: CameraConfig
    name: str = field(init=False)
    _capture: Any = field(init=False, default=None)
    _reader_thread: threading.Thread | None = field(init=False, default=None)
    _stop_event: threading.Event = field(init=False, default_factory=threading.Event)
    _frame_lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    _latest_frame: np.ndarray | None = field(init=False, default=None)
    _latest_timestamp_us: int = field(init=False, default=0)
    _delivered_timestamp_us: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.name = self.config.name
        if self.config.device is None and self.config.serial_number is None:
            raise CameraUnavailable(
                f"V4L2 camera {self.name} must define device or serial_number"
            )
        if self.config.depth:
            raise CameraUnavailable(f"V4L2 camera {self.name} does not support depth=true")

    def start(self) -> None:
        if self._capture is not None:
            return
        cv2 = _import_cv2()
        device = (
            self.config.device
            if self.config.device is not None
            else _resolve_v4l2_device_by_serial(str(self.config.serial_number))
        )
        open_deadline = time.monotonic() + self.config.startup_timeout_s
        capture = None
        attempt = 0
        while time.monotonic() < open_deadline:
            attempt += 1
            candidate = cv2.VideoCapture(device, cv2.CAP_V4L2)
            if candidate.isOpened():
                capture = candidate
                break
            candidate.release()
            if attempt == 1:
                log.warning(
                    "V4L2 camera %s could not open device %r; retrying for %.1fs",
                    self.name,
                    device,
                    self.config.startup_timeout_s,
                )
            time.sleep(0.2)
        if capture is None:
            raise CameraUnavailable(f"failed to open V4L2 device {device!r} for camera {self.name}")
        self._capture = capture
        self._configure_capture(cv2, capture)
        self._stop_event.clear()
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            args=(cv2,),
            name=f"v4l2-{self.name}",
            daemon=True,
        )
        self._reader_thread.start()
        deadline = time.monotonic() + self.config.startup_timeout_s
        while time.monotonic() < deadline:
            with self._frame_lock:
                if self._latest_frame is not None:
                    first_frame = self._latest_frame
                    break
            time.sleep(0.02)
        else:
            self.stop()
            raise CameraUnavailable(
                f"V4L2 camera {self.name} on {device!r} did not produce a frame within "
                f"{self.config.startup_timeout_s:.1f}s"
            )
        log.info(
            "V4L2 camera ready name=%s device=%s requested=%dx%d@%.1f format=%s "
            "actual=%dx%d@%.1f output=%s",
            self.name,
            device,
            self.config.width,
            self.config.height,
            self.config.fps,
            self.config.pixel_format,
            int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
            int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            capture.get(cv2.CAP_PROP_FPS),
            first_frame.shape,
        )

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._reader_thread
        if thread is not None:
            thread.join(timeout=1.0)
        capture = self._capture
        if capture is not None:
            capture.release()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._reader_thread = None
        self._capture = None

    def poll(self) -> CameraFrame | None:
        with self._frame_lock:
            if (
                self._latest_frame is None
                or self._latest_timestamp_us <= self._delivered_timestamp_us
            ):
                return None
            timestamp_us = self._latest_timestamp_us
            frame = self._latest_frame.copy()
            self._delivered_timestamp_us = timestamp_us
        return CameraFrame(self.name, timestamp_us, frame)

    def _configure_capture(self, cv2, capture) -> None:
        fourcc = cv2.VideoWriter_fourcc(*self.config.pixel_format)
        settings = (
            (cv2.CAP_PROP_FOURCC, fourcc, "pixel format"),
            (cv2.CAP_PROP_FRAME_WIDTH, self.config.width, "width"),
            (cv2.CAP_PROP_FRAME_HEIGHT, self.config.height, "height"),
            (cv2.CAP_PROP_FPS, self.config.fps, "fps"),
            (cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size, "buffer size"),
        )
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            settings += (
                (
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                    max(100.0, self.config.startup_timeout_s * 1000.0),
                    "read timeout",
                ),
            )
        for property_id, value, label in settings:
            if not capture.set(property_id, value):
                log.debug("V4L2 camera %s did not accept %s=%s", self.name, label, value)
        if self.config.exposure is not None:
            capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            if not capture.set(cv2.CAP_PROP_EXPOSURE, float(self.config.exposure)):
                log.warning(
                    "V4L2 camera %s did not accept exposure=%s; keeping device value",
                    self.name,
                    self.config.exposure,
                )

    def _reader_loop(self, cv2) -> None:
        consecutive_failures = 0
        warned = False
        while not self._stop_event.is_set():
            capture = self._capture
            if capture is None:
                return
            ok, frame = capture.read()
            if not ok or frame is None or not np.asarray(frame).size:
                consecutive_failures += 1
                if consecutive_failures >= 30 and not warned:
                    log.warning(
                        "V4L2 camera %s has %d consecutive read failures",
                        self.name,
                        consecutive_failures,
                    )
                    warned = True
                time.sleep(0.01)
                continue
            consecutive_failures = 0
            warned = False
            try:
                prepared = _prepare_v4l2_frame(frame, self.config, cv2)
            except Exception:
                log.exception("V4L2 frame preprocessing failed for camera %s", self.name)
                self._stop_event.set()
                return
            self._store_frame(prepared, now_us())

    def _store_frame(self, frame: np.ndarray, timestamp_us: int) -> None:
        with self._frame_lock:
            self._latest_frame = frame
            self._latest_timestamp_us = int(timestamp_us)


def _resolve_v4l2_device_by_serial(
    serial_number: str,
    by_id_directory: str | Path = "/dev/v4l/by-id",
) -> str:
    serial_number = str(serial_number).strip()
    if not serial_number:
        raise CameraUnavailable("V4L2 camera serial_number must be non-empty")
    directory = Path(by_id_directory)
    if not directory.is_dir():
        raise CameraUnavailable(f"V4L2 stable-device directory does not exist: {directory}")
    matches = sorted(
        path
        for path in directory.iterdir()
        if serial_number in path.name and path.name.endswith("-video-index0") and path.exists()
    )
    if not matches:
        raise CameraUnavailable(
            f"No V4L2 capture device found for serial_number={serial_number!r} in {directory}"
        )
    if len(matches) > 1:
        raise CameraUnavailable(
            f"Multiple V4L2 capture devices match serial_number={serial_number!r}: {matches}"
        )
    return str(matches[0])


class CameraManager:
    def __init__(self, cameras: list[CameraSource]) -> None:
        self.cameras = cameras

    @classmethod
    def from_config(cls, configs: tuple[CameraConfig, ...]) -> "CameraManager":
        cameras: list[CameraSource] = []
        for camera_config in configs:
            try:
                camera = _build_camera(camera_config)
            except CameraUnavailable as exc:
                log.warning("skip camera %s: %s", camera_config.name, exc)
                continue
            cameras.append(camera)
        return cls(cameras)

    def start(self) -> None:
        started: list[CameraSource] = []
        try:
            for camera in self.cameras:
                log.info("starting camera %s", camera.name)
                camera.start()
                started.append(camera)
        except Exception:
            for camera in reversed(started):
                camera.stop()
            raise

    def stop(self) -> None:
        for camera in self.cameras:
            try:
                camera.stop()
            except Exception as exc:  # pragma: no cover - shutdown guard
                log.debug("camera stop failed for %s: %s", camera.name, exc)

    def poll(self) -> list[CameraFrame]:
        frames: list[CameraFrame] = []
        for camera in self.cameras:
            frame = camera.poll()
            if frame is not None:
                frames.append(frame)
        return frames


def _build_camera(config: CameraConfig) -> CameraSource:
    backend = config.backend.lower().replace("-", "_")
    if backend in {"mock", "simulation"}:
        return MockCamera(config)
    if backend in {"orbbec", "orbbec_dabai", "orbbec_dabai_sdk"}:
        return OrbbecDabaiCamera(config)
    if backend in {"v4l2", "opencv_v4l2", "opencv"}:
        return V4L2Camera(config)
    raise CameraUnavailable(f"unsupported camera backend {config.backend!r}")


def _import_orbbec_module() -> object | None:
    for module_name in ("Orbbec_DaBai_SDK", "orbbec_dabai_sdk", "pyorbbecsdk"):
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue
    return None


def _import_cv2():
    try:
        return importlib.import_module("cv2")
    except ImportError as exc:
        raise CameraUnavailable(
            "V4L2 camera backend requires OpenCV; install opencv-python-headless>=4.8"
        ) from exc


def _prepare_v4l2_frame(frame: np.ndarray, config: CameraConfig, cv2) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise RuntimeError(
            f"V4L2 camera {config.name} returned unsupported frame shape {frame.shape}"
        )
    height, width = frame.shape[:2]
    y0, y1, x0, x1 = config.crop
    y0 = 0 if y0 is None else y0
    y1 = height if y1 is None else y1
    x0 = 0 if x0 is None else x0
    x1 = width if x1 is None else x1
    if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
        raise RuntimeError(
            f"camera {config.name} crop {config.crop} is outside frame {width}x{height}"
        )
    frame = frame[y0:y1, x0:x1]
    if config.output_size is not None:
        output_width, output_height = config.output_size
        shrinking = output_width < frame.shape[1] or output_height < frame.shape[0]
        interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
        frame = cv2.resize(frame, (output_width, output_height), interpolation=interpolation)
    # OpenCV V4L2 decodes to BGR; datasets use conventional RGB channel order.
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(frame, dtype=np.uint8)
