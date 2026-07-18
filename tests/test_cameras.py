from __future__ import annotations

from importlib.util import find_spec

import numpy as np
import pytest

from nero_collection.cameras import CameraManager, V4L2Camera, _build_camera, _prepare_v4l2_frame
from nero_collection.config import CameraConfig, _parse_camera


def test_parse_v4l2_camera_config() -> None:
    config = _parse_camera(
        {
            "name": "wrist",
            "backend": "v4l2",
            "device": "/dev/video2",
            "pixel_format": "mjpg",
            "buffer_size": 1,
            "startup_timeout_s": 2.5,
            "width": 640,
            "height": 480,
            "fps": 30,
            "crop": [10, 470, 20, 620],
            "output_size": [256, 192],
        }
    )

    assert config.device == "/dev/video2"
    assert config.pixel_format == "MJPG"
    assert config.buffer_size == 1
    assert config.startup_timeout_s == pytest.approx(2.5)
    assert config.crop == (10, 470, 20, 620)
    assert config.output_size == (256, 192)
    assert isinstance(_build_camera(config), V4L2Camera)


@pytest.mark.parametrize(
    "data",
    [
        {"name": "camera", "backend": "v4l2"},
        {"name": "camera", "backend": "v4l2", "device": "/dev/video2", "fps": 0},
        {"name": "camera", "backend": "v4l2", "device": "/dev/video2", "pixel_format": "MJPEG"},
        {"name": "camera", "backend": "v4l2", "device": "/dev/video2", "crop": [5, 4, 0, None]},
        {"name": "camera", "backend": "v4l2", "device": "/dev/video2", "output_size": [0, 192]},
    ],
)
def test_parse_v4l2_camera_rejects_invalid_settings(data) -> None:
    with pytest.raises(ValueError):
        _parse_camera(data)


@pytest.mark.skipif(find_spec("cv2") is None, reason="OpenCV is not installed")
def test_v4l2_preprocessing_crops_resizes_and_converts_to_rgb() -> None:
    import cv2

    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    frame[..., 0] = 10
    frame[..., 1] = 20
    frame[..., 2] = 30
    config = CameraConfig(
        name="camera",
        backend="v4l2",
        device="/dev/video2",
        width=6,
        height=4,
        crop=(1, 3, 2, 6),
        output_size=(2, 1),
    )

    output = _prepare_v4l2_frame(frame, config, cv2)

    assert output.shape == (1, 2, 3)
    assert output.dtype == np.uint8
    assert output.flags.c_contiguous
    assert np.all(output == np.asarray([30, 20, 10], dtype=np.uint8))


def test_v4l2_poll_returns_each_latest_frame_once() -> None:
    camera = V4L2Camera(CameraConfig(name="camera", backend="v4l2", device="/dev/video2"))
    first = np.full((2, 3, 3), 1, dtype=np.uint8)
    second = np.full((2, 3, 3), 2, dtype=np.uint8)

    camera._store_frame(first, 100)
    frame = camera.poll()
    assert frame is not None
    assert frame.timestamp_us == 100
    assert np.array_equal(frame.frame, first)
    assert camera.poll() is None

    camera._store_frame(second, 200)
    frame = camera.poll()
    assert frame is not None
    assert frame.timestamp_us == 200
    assert np.array_equal(frame.frame, second)


def test_camera_manager_stops_started_sources_after_start_failure() -> None:
    events: list[str] = []

    class Source:
        def __init__(self, name: str, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        def start(self) -> None:
            events.append(f"{self.name}:start")
            if self.fail:
                raise RuntimeError("start failed")

        def stop(self) -> None:
            events.append(f"{self.name}:stop")

    manager = CameraManager([Source("first"), Source("second", fail=True)])

    with pytest.raises(RuntimeError, match="start failed"):
        manager.start()

    assert events == ["first:start", "second:start", "first:stop"]
