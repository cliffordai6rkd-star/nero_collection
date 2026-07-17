from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nero_collection.config import CollectionConfig
from nero_collection.filters import DatasetFilterBank, LowPassVelocityDifferentiator
from nero_collection.time_utils import now_us


FORMAT_VERSION = "factr_multimodal_episode/v2"


@dataclass
class EpisodeBuffer:
    config: CollectionConfig
    arm_names: tuple[str, ...]
    sample_rate_hz: float
    teleop_timestamps_us: list[int] = field(default_factory=list)
    teleop_data: dict[str, list[np.ndarray]] = field(default_factory=lambda: defaultdict(list))
    teleop_state_names: dict[str, str] = field(default_factory=dict)
    camera_frames: dict[str, list[np.ndarray]] = field(default_factory=lambda: defaultdict(list))
    camera_timestamps_us: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    acceleration_estimators: dict[str, LowPassVelocityDifferentiator] = field(
        init=False,
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        self.filter_bank = DatasetFilterBank(self.config.robot_states, self.sample_rate_hz)

    def append_teleop(self, timestamp_us: int, values: dict[str, tuple[str, np.ndarray]]) -> None:
        acceleration_values = {
            dataset_name: value
            for dataset_name, (state_name, value) in values.items()
            if state_name == "acceleration"
        }
        for dataset_name in acceleration_values:
            velocity_name = _velocity_dataset_name(dataset_name)
            if velocity_name not in values:
                raise KeyError(
                    f"cannot derive {dataset_name}: missing source velocity dataset {velocity_name}"
                )

        self.teleop_timestamps_us.append(int(timestamp_us))
        for dataset_name, (state_name, value) in values.items():
            if state_name == "acceleration":
                continue
            filtered = self.filter_bank.apply(dataset_name, state_name, np.asarray(value))
            self.teleop_data[dataset_name].append(filtered)
            self.teleop_state_names[dataset_name] = state_name

        acceleration_config = self.config.robot_states.get("acceleration")
        velocity_cutoff_hz = (
            acceleration_config.velocity_lowpass_cutoff_hz
            if acceleration_config is not None
            else None
        )
        for dataset_name in acceleration_values:
            velocity_name = _velocity_dataset_name(dataset_name)
            raw_velocity = np.asarray(values[velocity_name][1])
            estimator = self.acceleration_estimators.get(dataset_name)
            if estimator is None:
                estimator = LowPassVelocityDifferentiator(velocity_cutoff_hz)
                self.acceleration_estimators[dataset_name] = estimator
            acceleration = estimator.apply(raw_velocity, timestamp_us)
            acceleration = self.filter_bank.apply(
                dataset_name,
                "acceleration",
                acceleration,
            )
            self.teleop_data[dataset_name].append(acceleration)
            self.teleop_state_names[dataset_name] = "acceleration"

    def append_camera(self, camera_name: str, timestamp_us: int, frame: np.ndarray) -> None:
        self.camera_timestamps_us[camera_name].append(int(timestamp_us))
        self.camera_frames[camera_name].append(np.asarray(frame, dtype=np.uint8))

    @property
    def sample_count(self) -> int:
        return len(self.teleop_timestamps_us)

    def save(self, path: str | Path) -> Path:
        try:
            import h5py
        except Exception as exc:
            raise RuntimeError(
                "Failed to import h5py. Reinstall compatible numpy/h5py versions, for example: "
                'python -m pip install --upgrade --force-reinstall "numpy>=1.23,<3" "h5py>=3.11"'
            ) from exc

        out_path = Path(path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        string_dtype = h5py.string_dtype(encoding="utf-8")

        with h5py.File(tmp_path, "w") as h5:
            h5.attrs["format"] = FORMAT_VERSION
            h5.attrs["saved_at_us"] = now_us()
            h5.create_dataset("config_yaml", data=self.config.raw_yaml, dtype=string_dtype)

            teleop = h5.create_group("teleop")
            teleop.attrs["arm_names"] = np.asarray(self.arm_names, dtype=string_dtype)
            teleop.attrs["joint_layout"] = "joint vectors are concatenated in arm_names order"
            teleop.attrs["pose_layout"] = "single arm: (N,4,4); multi arm: (N,A,4,4), A follows arm_names"
            teleop.create_dataset("timestamp_us", data=np.asarray(self.teleop_timestamps_us, dtype=np.int64))
            for name, values in sorted(self.teleop_data.items()):
                data = _stack(values)
                dataset = teleop.create_dataset(name, data=data, compression=_compression_for(data))
                state_name = self.teleop_state_names.get(name, "")
                dataset.attrs["state_name"] = state_name
                state_config = self.config.robot_states.get(state_name)
                dataset.attrs["lowpass"] = bool(state_config.lowpass) if state_config else False
                if state_name == "acceleration":
                    dataset.attrs["derived_from"] = _velocity_dataset_name(name)
                    dataset.attrs["derivative_method"] = "velocity_lowpass_then_backward_difference"
                    dataset.attrs["timestamp_path"] = "teleop/timestamp_us"
                    cutoff_hz = state_config.velocity_lowpass_cutoff_hz if state_config else None
                    dataset.attrs["velocity_lowpass"] = cutoff_hz is not None
                    if cutoff_hz is not None:
                        dataset.attrs["velocity_lowpass_cutoff_hz"] = cutoff_hz
                if name in {"ee_pose", "cmd_ee_pose", "ee_pose_leader"}:
                    dataset.attrs["frame_name"] = "tcp"
                    dataset.attrs["frame_type"] = "end_effector"

            if self.camera_frames:
                cameras = h5.create_group("cameras")
                for camera_name, frames in sorted(self.camera_frames.items()):
                    if not frames:
                        continue
                    group = cameras.create_group(camera_name)
                    stacked_frames = np.stack(frames, axis=0)
                    group.create_dataset("frames", data=stacked_frames, compression="gzip", compression_opts=4)
                    group.create_dataset(
                        "timestamp_us",
                        data=np.asarray(self.camera_timestamps_us[camera_name], dtype=np.int64),
                    )
                    group.attrs["timeline"] = f"cameras/{camera_name}/timestamp_us"

            meta = h5.create_group("metadata")
            meta.create_dataset("arm_names_json", data=json.dumps(list(self.arm_names)), dtype=string_dtype)

        tmp_path.replace(out_path)
        return out_path


def _stack(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        return np.empty((0,), dtype=np.float64)
    return np.stack(values, axis=0)


def _compression_for(data: np.ndarray) -> str | None:
    if data.dtype == np.uint8 or data.size > 2048:
        return "gzip"
    return None


def _velocity_dataset_name(acceleration_dataset_name: str) -> str:
    if acceleration_dataset_name == "ddq":
        return "dq"
    if acceleration_dataset_name.startswith("ddq_"):
        return f"dq_{acceleration_dataset_name.removeprefix('ddq_')}"
    raise ValueError(
        f"acceleration dataset name must be 'ddq' or start with 'ddq_': "
        f"{acceleration_dataset_name}"
    )
