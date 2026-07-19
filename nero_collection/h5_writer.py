from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nero_collection.config import CollectionConfig
from nero_collection.contact_wrench import PinocchioJointTorqueResidualEstimator
from nero_collection.dynamics_processing import (
    filter_torque,
    reconstruct_state_from_positions,
    resample_columns,
    select_source_timestamps,
)
from nero_collection.filters import DatasetFilterBank, LowPassVelocityDifferentiator
from nero_collection.time_utils import now_us


FORMAT_VERSION = "factr_multimodal_episode/v3"


@dataclass
class EpisodeBuffer:
    config: CollectionConfig
    arm_names: tuple[str, ...]
    sample_rate_hz: float
    teleop_timestamps_us: list[int] = field(default_factory=list)
    teleop_data: dict[str, list[np.ndarray]] = field(default_factory=lambda: defaultdict(list))
    teleop_state_names: dict[str, str] = field(default_factory=dict)
    teleop_raw_data: dict[str, list[np.ndarray]] = field(default_factory=lambda: defaultdict(list))
    teleop_raw_state_names: dict[str, str] = field(default_factory=dict)
    camera_frames: dict[str, list[np.ndarray]] = field(default_factory=lambda: defaultdict(list))
    camera_timestamps_us: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    acceleration_estimators: dict[str, LowPassVelocityDifferentiator] = field(
        init=False,
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        self.filter_bank = DatasetFilterBank(self.config.robot_states)

    def append_teleop(
        self,
        timestamp_us: int,
        values: dict[str, tuple[str, np.ndarray]],
    ) -> dict[str, tuple[str, np.ndarray]]:
        timestamp_us = int(timestamp_us)
        if self.teleop_timestamps_us:
            previous = self.teleop_timestamps_us[-1]
            if timestamp_us < previous:
                raise ValueError(
                    f"teleop observation timestamp moved backwards: {timestamp_us} < {previous}"
                )
            if timestamp_us == previous:
                return {}
        processed_values: dict[str, tuple[str, np.ndarray]] = {}
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

        self.teleop_timestamps_us.append(timestamp_us)
        for dataset_name, (state_name, value) in values.items():
            self.teleop_raw_data[dataset_name].append(np.asarray(value).copy())
            self.teleop_raw_state_names[dataset_name] = state_name
        for dataset_name, (state_name, value) in values.items():
            if state_name == "acceleration":
                continue
            filtered = self.filter_bank.apply(
                dataset_name,
                state_name,
                np.asarray(value),
                timestamp_us,
            )
            self.teleop_data[dataset_name].append(filtered)
            self.teleop_state_names[dataset_name] = state_name
            processed_values[dataset_name] = (state_name, np.asarray(filtered).copy())

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
                timestamp_us,
            )
            self.teleop_data[dataset_name].append(acceleration)
            self.teleop_state_names[dataset_name] = "acceleration"
            processed_values[dataset_name] = (
                "acceleration",
                np.asarray(acceleration).copy(),
            )
        return processed_values

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
        finalized_data, finalized_state_names, finalized_attrs = self._finalize_teleop_data()

        with h5py.File(tmp_path, "w") as h5:
            h5.attrs["format"] = FORMAT_VERSION
            h5.attrs["saved_at_us"] = now_us()
            h5.create_dataset("config_yaml", data=self.config.raw_yaml, dtype=string_dtype)

            teleop = h5.create_group("teleop")
            teleop.attrs["arm_names"] = np.asarray(self.arm_names, dtype=string_dtype)
            teleop.attrs["joint_layout"] = "joint vectors are concatenated in arm_names order"
            teleop.attrs["pose_layout"] = "single arm: (N,4,4); multi arm: (N,A,4,4), A follows arm_names"
            teleop_timestamp = teleop.create_dataset(
                "timestamp_us",
                data=np.asarray(self.teleop_timestamps_us, dtype=np.int64),
            )
            teleop_timestamp.attrs["source"] = "primary_follower_joint_observation"
            teleop_timestamp.attrs["clock"] = "unix_epoch"
            teleop_timestamp.attrs["unit"] = "us"
            for name, data in sorted(finalized_data.items()):
                dataset = teleop.create_dataset(name, data=data, compression=_compression_for(data))
                state_name = finalized_state_names.get(name, "")
                dataset.attrs["state_name"] = state_name
                state_config = self.config.robot_states.get(state_name)
                dataset.attrs["lowpass"] = bool(state_config.lowpass) if state_config else False
                dataset.attrs["median_window"] = state_config.median_window if state_config else 1
                if state_config and state_config.lowpass:
                    dataset.attrs["lowpass_cutoff_hz"] = state_config.lowpass_cutoff_hz
                    dataset.attrs["filter_timeline"] = "teleop/timestamp_us"
                if state_name == "acceleration" and "derivative_method" not in finalized_attrs.get(name, {}):
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
                if state_name == "timestamp":
                    dataset.attrs["clock"] = "unix_epoch"
                    dataset.attrs["unit"] = "us"
                for key, value in finalized_attrs.get(name, {}).items():
                    dataset.attrs[key] = value

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

    def _finalize_teleop_data(
        self,
    ) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, dict[str, object]]]:
        data = {name: _stack(values) for name, values in self.teleop_data.items()}
        state_names = dict(self.teleop_state_names)
        attrs: dict[str, dict[str, object]] = defaultdict(dict)
        processing = self.config.dynamics_processing
        if not processing.enabled:
            return data, state_names, attrs

        timeline = np.asarray(self.teleop_timestamps_us, dtype=np.int64)
        if timeline.size < processing.min_samples:
            raise RuntimeError(
                "Dynamics-aware episode saving requires at least "
                f"{processing.min_samples} samples; got {timeline.size}"
            )
        if np.any(np.diff(timeline) <= 0):
            raise RuntimeError("Teleop acquisition timestamps must be strictly increasing")

        raw = {name: _stack(values) for name, values in self.teleop_raw_data.items()}
        arm_count = len(self.arm_names)
        for role in ("leader", "follower"):
            q_name = f"q_{role}"
            dq_name = f"dq_{role}"
            ddq_name = f"ddq_{role}"
            q_timestamp_name = f"q_timestamp_{role}_us"
            q_acquired_timestamp_name = f"q_acquired_timestamp_{role}_us"
            if q_name not in raw:
                continue
            q_raw = np.asarray(raw[q_name], dtype=np.float64)
            if q_raw.ndim != 2 or q_raw.shape[1] % arm_count != 0:
                raise RuntimeError(f"Cannot postprocess {q_name} with shape {q_raw.shape}")
            dof = q_raw.shape[1] // arm_count
            q_timestamp, q_timestamp_fallback = _effective_arm_timestamp_matrix(
                raw.get(q_timestamp_name),
                raw.get(q_acquired_timestamp_name),
                timeline,
                arm_count,
                q_timestamp_name,
            )
            q_parts: list[np.ndarray] = []
            dq_parts: list[np.ndarray] = []
            ddq_parts: list[np.ndarray] = []
            for arm_index in range(arm_count):
                start = arm_index * dof
                stop = start + dof
                q_fit, dq, ddq = reconstruct_state_from_positions(
                    q_timestamp[:, arm_index],
                    q_raw[:, start:stop],
                    state_method=processing.state_method,
                    spline_smoothing_rad2=processing.spline_smoothing_rad2,
                    fourier_fundamental_hz=processing.fourier_fundamental_hz,
                    fourier_harmonics=processing.fourier_harmonics,
                    evaluation_timestamp_us=timeline,
                )
                q_parts.append(q_fit)
                dq_parts.append(dq)
                ddq_parts.append(ddq)
            data[q_name] = np.concatenate(q_parts, axis=1)
            if dq_name in data:
                data[dq_name] = np.concatenate(dq_parts, axis=1)
            if ddq_name in data:
                data[ddq_name] = np.concatenate(ddq_parts, axis=1)
            _add_raw_dataset(data, state_names, attrs, f"{q_name}_raw", q_raw, "q")
            if dq_name in raw:
                _add_raw_dataset(
                    data,
                    state_names,
                    attrs,
                    f"{dq_name}_firmware_raw",
                    raw[dq_name],
                    "velocity",
                )
            if ddq_name in raw:
                _add_raw_dataset(
                    data,
                    state_names,
                    attrs,
                    f"{ddq_name}_adapter_raw",
                    raw[ddq_name],
                    "acceleration",
                )
            attrs[q_name].update(
                {
                    "processing_method": processing.state_method,
                    "source_dataset": f"{q_name}_raw",
                    "source_timestamp_path": f"teleop/{q_timestamp_name}",
                    "fallback_timestamp_path": f"teleop/{q_acquired_timestamp_name}",
                    "timestamp_fallback_arms": np.asarray(
                        q_timestamp_fallback,
                        dtype=np.uint8,
                    ),
                    "resampled_to": "teleop/timestamp_us",
                    "spline_smoothing_rad2": processing.spline_smoothing_rad2,
                    "lowpass": False,
                }
            )
            if dq_name in data:
                attrs[dq_name].update(
                    {
                        "derived_from": q_name,
                        "derivative_method": f"{processing.state_method}_analytic_first_derivative",
                        "timestamp_path": "teleop/timestamp_us",
                        "firmware_source_dataset": f"{dq_name}_firmware_raw",
                        "lowpass": False,
                    }
                )
            if ddq_name in data:
                attrs[ddq_name].update(
                    {
                        "derived_from": q_name,
                        "derivative_method": f"{processing.state_method}_analytic_second_derivative",
                        "timestamp_path": "teleop/timestamp_us",
                        "adapter_source_dataset": f"{ddq_name}_adapter_raw",
                        "lowpass": False,
                    }
                )

            tau_name = f"tau_{role}"
            motor_timestamp_name = f"motor_timestamp_{role}_us"
            motor_acquired_timestamp_name = f"motor_acquired_timestamp_{role}_us"
            if tau_name not in raw:
                continue
            tau_raw = np.asarray(raw[tau_name], dtype=np.float64)
            motor_timestamp = _joint_timestamp_matrix(
                raw.get(motor_timestamp_name),
                np.zeros_like(timeline),
                tau_raw.shape,
                motor_timestamp_name,
            )
            motor_acquired_timestamp = _joint_timestamp_matrix(
                raw.get(motor_acquired_timestamp_name),
                timeline,
                tau_raw.shape,
                motor_acquired_timestamp_name,
            )
            tau_aligned = resample_columns(
                motor_timestamp,
                tau_raw,
                timeline,
                fallback_source_timestamp_us=motor_acquired_timestamp,
            )
            data[tau_name] = filter_torque(
                timeline,
                tau_aligned,
                median_window=processing.torque_median_window,
                lowpass_hz=processing.torque_lowpass_hz,
            )
            _add_raw_dataset(data, state_names, attrs, f"{tau_name}_raw", tau_raw, "torque")
            attrs[tau_name].update(
                {
                    "processing_method": "median_then_fourth_order_zero_phase_butterworth",
                    "source_dataset": f"{tau_name}_raw",
                    "source_timestamp_path": f"teleop/{motor_timestamp_name}",
                    "resampled_to": "teleop/timestamp_us",
                    "median_window": processing.torque_median_window,
                    "lowpass_cutoff_hz": processing.torque_lowpass_hz,
                    "zero_phase": True,
                    "fallback_timestamp_path": f"teleop/{motor_acquired_timestamp_name}",
                }
            )
        self._append_follower_dynamics_estimates(data, state_names, attrs)
        return data, state_names, attrs

    def _append_follower_dynamics_estimates(self, data, state_names, attrs) -> None:
        required = ("q_follower", "dq_follower", "ddq_follower", "tau_follower")
        if not all(name in data for name in required):
            return
        if any(np.asarray(data[name]).shape[1:] != (7,) for name in required):
            return
        estimator = PinocchioJointTorqueResidualEstimator(
            self.config.realtime_plot.inverse_dynamics
        )
        estimates = [
            estimator.estimate(q, dq, ddq, tau)
            for q, dq, ddq, tau in zip(
                data["q_follower"],
                data["dq_follower"],
                data["ddq_follower"],
                data["tau_follower"],
            )
        ]
        outputs = {
            "tau_id_follower": np.stack([estimate.tau_id for estimate in estimates]),
            "tau_friction_follower": np.stack(
                [estimate.tau_friction for estimate in estimates]
            ),
            "tau_bias_follower": np.stack([estimate.tau_bias for estimate in estimates]),
            "tau_model_follower": np.stack([estimate.tau_model for estimate in estimates]),
            "tau_ext_follower": np.stack([estimate.tau_residual for estimate in estimates]),
        }
        for name, value in outputs.items():
            data[name] = value
            state_names[name] = "torque"
            attrs[name].update(
                {
                    "timestamp_path": "teleop/timestamp_us",
                    "lowpass": False,
                    "model_urdf": str(self.config.realtime_plot.inverse_dynamics.urdf_path),
                    "model_manifest": str(
                        self.config.realtime_plot.inverse_dynamics.manifest_path or ""
                    ),
                }
            )
        attrs["tau_ext_follower"]["definition"] = "tau_model_follower - tau_follower"


def _stack(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        return np.empty((0,), dtype=np.float64)
    return np.stack(values, axis=0)


def _effective_arm_timestamp_matrix(
    primary_value: np.ndarray | None,
    acquired_value: np.ndarray | None,
    fallback: np.ndarray,
    arm_count: int,
    name: str,
) -> tuple[np.ndarray, list[bool]]:
    expected_shape = (fallback.size, arm_count)
    primary = _timestamp_array(primary_value, expected_shape, np.zeros(expected_shape), name)
    acquired = _timestamp_array(
        acquired_value,
        expected_shape,
        np.repeat(fallback[:, None], arm_count, axis=1),
        name.replace("q_timestamp", "q_acquired_timestamp"),
    )
    result = np.empty(expected_shape, dtype=np.int64)
    used_fallback: list[bool] = []
    for arm_index in range(arm_count):
        selected, fallback_used = select_source_timestamps(
            primary[:, arm_index],
            acquired[:, arm_index],
            minimum_unique=4,
        )
        result[:, arm_index] = selected
        used_fallback.append(fallback_used)
    return result, used_fallback


def _timestamp_array(value, expected_shape, fallback, name):
    if value is None:
        return np.asarray(fallback, dtype=np.int64).copy()
    timestamp = np.asarray(value, dtype=np.int64)
    if timestamp.shape == (expected_shape[0],) and expected_shape[1] == 1:
        timestamp = timestamp[:, None]
    if timestamp.shape != expected_shape:
        raise RuntimeError(f"{name} must have shape {expected_shape}; got {timestamp.shape}")
    result = timestamp.copy()
    invalid = result <= 0
    fallback_array = np.asarray(fallback, dtype=np.int64)
    result[invalid] = fallback_array[invalid]
    return result


def _joint_timestamp_matrix(
    value: np.ndarray | None,
    fallback: np.ndarray,
    expected_shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    if len(expected_shape) != 2 or expected_shape[0] != fallback.size:
        raise RuntimeError(f"Invalid measured joint data shape for {name}: {expected_shape}")
    fallback_matrix = np.repeat(fallback[:, None], expected_shape[1], axis=1)
    if value is None:
        return fallback_matrix
    timestamp = np.asarray(value, dtype=np.int64)
    if timestamp.shape != expected_shape:
        raise RuntimeError(f"{name} must have shape {expected_shape}; got {timestamp.shape}")
    result = timestamp.copy()
    invalid = result <= 0
    result[invalid] = fallback_matrix[invalid]
    return result


def _add_raw_dataset(data, state_names, attrs, name, value, state_name):
    data[name] = np.asarray(value).copy()
    state_names[name] = state_name
    attrs[name].update(
        {
            "raw": True,
            "filter_applied": False,
            "lowpass": False,
            "median_window": 1,
        }
    )


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
