from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass, field

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
        for camera in self.cameras:
            log.info("starting camera %s", camera.name)
            camera.start()

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
    raise CameraUnavailable(f"unsupported camera backend {config.backend!r}")


def _import_orbbec_module() -> object | None:
    for module_name in ("Orbbec_DaBai_SDK", "orbbec_dabai_sdk", "pyorbbecsdk"):
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue
    return None
