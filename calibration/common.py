from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class CalibrationPose:
    name: str
    q: np.ndarray


@dataclass(frozen=True)
class MotionSettings:
    command_rate_hz: float
    joint_speed_rad_s: float
    max_step_rad: float
    max_motion_velocity_rad_s: float
    motion_timeout_s: float
    settle_timeout_s: float
    settle_duration_s: float
    sample_rate_hz: float
    sample_duration_s: float
    max_static_velocity_rad_s: float
    max_position_error_rad: float
    round_count: int
    alternate_reverse: bool


@dataclass(frozen=True)
class SafetySettings:
    approved: bool
    joint_limit_margin_rad: float
    max_abs_torque_nm: np.ndarray


@dataclass(frozen=True)
class ModelSettings:
    urdf_path: Path
    locked_joint_names: tuple[str, ...]
    terminal_joint_name: str
    gravity_m_s2: np.ndarray


@dataclass(frozen=True)
class FitSettings:
    min_poses: int
    max_condition_number: float
    mass_bounds_kg: tuple[float, float]
    max_abs_com_m: float
    max_abs_bias_nm: np.ndarray
    holdout_fraction: float
    split_seed: int
    min_holdout_improvement_ratio: float
    max_round_mass_relative_range: float
    max_round_com_spread_m: float
    max_round_bias_range_nm: np.ndarray


@dataclass(frozen=True)
class CalibrationPlan:
    source_path: Path
    collection_config_path: Path
    pair_name: str
    poses: tuple[CalibrationPose, ...]
    motion: MotionSettings
    safety: SafetySettings
    model: ModelSettings
    fit: FitSettings


@dataclass(frozen=True)
class StaticDataset:
    q: np.ndarray
    tau: np.ndarray
    current: np.ndarray
    timestamp_us: np.ndarray
    pose_index: np.ndarray
    round_index: np.ndarray
    pose_names: tuple[str, ...]
    metadata: dict[str, Any]


def load_plan(path: str | Path) -> CalibrationPlan:
    source_path = Path(path).expanduser().resolve()
    data = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("calibration config must be a YAML mapping")
    base_dir = source_path.parent

    collection_config_path = _resolve_path(
        data.get("collection_config", "../configs/master_slave_can.yaml"), base_dir
    )
    pair_name = str(data.get("pair", "main"))

    pose_items = data.get("poses", [])
    if not isinstance(pose_items, list) or not pose_items:
        raise ValueError("calibration poses must be a non-empty list")
    poses: list[CalibrationPose] = []
    seen_names: set[str] = set()
    for index, item in enumerate(pose_items):
        if not isinstance(item, dict):
            raise ValueError(f"poses[{index}] must be a mapping")
        name = str(item.get("name", f"pose_{index:02d}"))
        if name in seen_names:
            raise ValueError(f"duplicate calibration pose name: {name}")
        seen_names.add(name)
        poses.append(CalibrationPose(name=name, q=_vector(item.get("q"), 7, f"poses[{index}].q")))

    motion_data = _mapping(data.get("motion", {}), "motion")
    motion = MotionSettings(
        command_rate_hz=_positive(motion_data.get("command_rate_hz", 30.0), "motion.command_rate_hz"),
        joint_speed_rad_s=_positive(motion_data.get("joint_speed_rad_s", 0.12), "motion.joint_speed_rad_s"),
        max_step_rad=_positive(motion_data.get("max_step_rad", 0.01), "motion.max_step_rad"),
        max_motion_velocity_rad_s=_positive(
            motion_data.get("max_motion_velocity_rad_s", 0.4),
            "motion.max_motion_velocity_rad_s",
        ),
        motion_timeout_s=_positive(motion_data.get("motion_timeout_s", 20.0), "motion.motion_timeout_s"),
        settle_timeout_s=_positive(motion_data.get("settle_timeout_s", 8.0), "motion.settle_timeout_s"),
        settle_duration_s=_positive(motion_data.get("settle_duration_s", 1.0), "motion.settle_duration_s"),
        sample_rate_hz=_positive(motion_data.get("sample_rate_hz", 100.0), "motion.sample_rate_hz"),
        sample_duration_s=_positive(motion_data.get("sample_duration_s", 2.0), "motion.sample_duration_s"),
        max_static_velocity_rad_s=_positive(
            motion_data.get("max_static_velocity_rad_s", 0.02),
            "motion.max_static_velocity_rad_s",
        ),
        max_position_error_rad=_positive(
            motion_data.get("max_position_error_rad", 0.02),
            "motion.max_position_error_rad",
        ),
        round_count=int(motion_data.get("round_count", 3)),
        alternate_reverse=bool(motion_data.get("alternate_reverse", True)),
    )
    if motion.max_motion_velocity_rad_s <= motion.joint_speed_rad_s:
        raise ValueError(
            "motion.max_motion_velocity_rad_s must be greater than motion.joint_speed_rad_s"
        )
    if motion.round_count < 1:
        raise ValueError("motion.round_count must be at least 1")

    safety_data = _mapping(data.get("safety", {}), "safety")
    safety = SafetySettings(
        approved=bool(safety_data.get("approved", False)),
        joint_limit_margin_rad=_non_negative(
            safety_data.get("joint_limit_margin_rad", 0.05),
            "safety.joint_limit_margin_rad",
        ),
        max_abs_torque_nm=_positive_vector(
            safety_data.get("max_abs_torque_nm", [20.0, 20.0, 12.0, 12.0, 6.0, 6.0, 6.0]),
            7,
            "safety.max_abs_torque_nm",
        ),
    )

    model_data = _mapping(data.get("model", {}), "model")
    model = ModelSettings(
        urdf_path=_resolve_path(
            model_data.get("urdf_path", "../urdf/nero/nero_with_gripper.urdf"),
            base_dir,
        ),
        locked_joint_names=tuple(
            str(value)
            for value in model_data.get(
                "locked_joint_names", ["gripper", "gripper_joint1", "gripper_joint2"]
            )
        ),
        terminal_joint_name=str(model_data.get("terminal_joint_name", "joint7")),
        gravity_m_s2=_vector(
            model_data.get("gravity_m_s2", [0.0, 0.0, -9.81]), 3, "model.gravity_m_s2"
        ),
    )
    if not model.urdf_path.is_file():
        raise ValueError(f"URDF does not exist: {model.urdf_path}")

    fit_data = _mapping(data.get("fit", {}), "fit")
    mass_bounds = _vector(fit_data.get("mass_bounds_kg", [0.05, 2.0]), 2, "fit.mass_bounds_kg")
    if mass_bounds[0] < 0 or mass_bounds[1] <= mass_bounds[0]:
        raise ValueError("fit.mass_bounds_kg must be [non-negative min, larger max]")
    fit = FitSettings(
        min_poses=int(fit_data.get("min_poses", 10)),
        max_condition_number=_positive(
            fit_data.get("max_condition_number", 1.0e8), "fit.max_condition_number"
        ),
        mass_bounds_kg=(float(mass_bounds[0]), float(mass_bounds[1])),
        max_abs_com_m=_positive(fit_data.get("max_abs_com_m", 0.30), "fit.max_abs_com_m"),
        max_abs_bias_nm=_positive_vector(
            fit_data.get("max_abs_bias_nm", [8.0, 8.0, 6.0, 6.0, 4.0, 4.0, 4.0]),
            7,
            "fit.max_abs_bias_nm",
        ),
        holdout_fraction=float(fit_data.get("holdout_fraction", 0.30)),
        split_seed=int(fit_data.get("split_seed", 20260718)),
        min_holdout_improvement_ratio=float(
            fit_data.get("min_holdout_improvement_ratio", 0.30)
        ),
        max_round_mass_relative_range=_positive(
            fit_data.get("max_round_mass_relative_range", 0.05),
            "fit.max_round_mass_relative_range",
        ),
        max_round_com_spread_m=_positive(
            fit_data.get("max_round_com_spread_m", 0.01),
            "fit.max_round_com_spread_m",
        ),
        max_round_bias_range_nm=_positive_vector(
            fit_data.get("max_round_bias_range_nm", [0.4, 0.4, 0.3, 0.3, 0.2, 0.2, 0.2]),
            7,
            "fit.max_round_bias_range_nm",
        ),
    )
    if fit.min_poses < 5:
        raise ValueError("fit.min_poses must be at least 5")
    if not 0.1 <= fit.holdout_fraction <= 0.5:
        raise ValueError("fit.holdout_fraction must be within [0.1, 0.5]")
    if not 0.0 <= fit.min_holdout_improvement_ratio < 1.0:
        raise ValueError("fit.min_holdout_improvement_ratio must be within [0, 1)")
    return CalibrationPlan(
        source_path=source_path,
        collection_config_path=collection_config_path,
        pair_name=pair_name,
        poses=tuple(poses),
        motion=motion,
        safety=safety,
        model=model,
        fit=fit,
    )


def save_dataset(path: str | Path, dataset: StaticDataset) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        q=np.asarray(dataset.q, dtype=np.float64),
        tau=np.asarray(dataset.tau, dtype=np.float64),
        current=np.asarray(dataset.current, dtype=np.float64),
        timestamp_us=np.asarray(dataset.timestamp_us, dtype=np.int64),
        pose_index=np.asarray(dataset.pose_index, dtype=np.int32),
        round_index=np.asarray(dataset.round_index, dtype=np.int32),
        pose_names=np.asarray(dataset.pose_names, dtype=np.str_),
        metadata_json=np.asarray(json.dumps(dataset.metadata, sort_keys=True)),
    )
    return output


def load_dataset(path: str | Path) -> StaticDataset:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as values:
        required = {"q", "tau", "current", "timestamp_us", "pose_index", "pose_names", "metadata_json"}
        missing = sorted(required.difference(values.files))
        if missing:
            raise ValueError(f"calibration dataset is missing arrays: {missing}")
        dataset = StaticDataset(
            q=np.asarray(values["q"], dtype=np.float64),
            tau=np.asarray(values["tau"], dtype=np.float64),
            current=np.asarray(values["current"], dtype=np.float64),
            timestamp_us=np.asarray(values["timestamp_us"], dtype=np.int64),
            pose_index=np.asarray(values["pose_index"], dtype=np.int32),
            round_index=(
                np.asarray(values["round_index"], dtype=np.int32)
                if "round_index" in values.files
                else np.zeros(np.asarray(values["pose_index"]).shape, dtype=np.int32)
            ),
            pose_names=tuple(str(value) for value in values["pose_names"].tolist()),
            metadata=json.loads(str(values["metadata_json"].item())),
        )
    _validate_dataset(dataset)
    return dataset


def aggregate_static_poses(
    dataset: StaticDataset,
    round_index: int | None = None,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    names: list[str] = []
    q_values: list[np.ndarray] = []
    tau_values: list[np.ndarray] = []
    tau_std_values: list[np.ndarray] = []
    for index, name in enumerate(dataset.pose_names):
        mask = dataset.pose_index == index
        if round_index is not None:
            mask &= dataset.round_index == int(round_index)
        if not np.any(mask):
            continue
        names.append(name)
        q_values.append(np.median(dataset.q[mask], axis=0))
        tau_values.append(np.median(dataset.tau[mask], axis=0))
        tau_std_values.append(np.std(dataset.tau[mask], axis=0, ddof=0))
    if not names:
        raise ValueError("calibration dataset contains no populated poses")
    return (
        tuple(names),
        np.stack(q_values),
        np.stack(tau_values),
        np.stack(tau_std_values),
    )


def setup_socketcan(channel: str, bitrate: int) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    script = repository_root / "scripts" / "setup_can.sh"
    environment = os.environ.copy()
    environment["CAN_BITRATE"] = str(int(bitrate))
    subprocess.run(
        ["bash", str(script), str(channel)],
        cwd=repository_root,
        env=environment,
        check=True,
    )


def metadata_for_plan(plan: CalibrationPlan) -> dict[str, Any]:
    return {
        "calibration_config": str(plan.source_path),
        "collection_config": str(plan.collection_config_path),
        "pair": plan.pair_name,
        "urdf_path": str(plan.model.urdf_path),
        "terminal_joint_name": plan.model.terminal_joint_name,
        "gravity_m_s2": plan.model.gravity_m_s2.tolist(),
    }


def _validate_dataset(dataset: StaticDataset) -> None:
    sample_count = dataset.q.shape[0]
    for name, values, width in (
        ("q", dataset.q, 7),
        ("tau", dataset.tau, 7),
        ("current", dataset.current, 7),
    ):
        if values.shape != (sample_count, width):
            raise ValueError(f"{name} must have shape (N, {width}); got {values.shape}")
    if (
        dataset.timestamp_us.shape != (sample_count,)
        or dataset.pose_index.shape != (sample_count,)
        or dataset.round_index.shape != (sample_count,)
    ):
        raise ValueError("timestamp_us, pose_index, and round_index must have shape (N,)")
    if not np.isfinite(dataset.q).all() or not np.isfinite(dataset.tau).all():
        raise ValueError("q and tau calibration samples must be finite")
    if np.any(dataset.pose_index < 0) or np.any(dataset.pose_index >= len(dataset.pose_names)):
        raise ValueError("pose_index contains an invalid pose id")
    if np.any(dataset.round_index < 0):
        raise ValueError("round_index must be non-negative")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size != size or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite {size}D vector")
    return vector


def _positive_vector(value: Any, size: int, name: str) -> np.ndarray:
    vector = _vector(value, size, name)
    if np.any(vector <= 0):
        raise ValueError(f"{name} values must be positive")
    return vector


def _positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    result = float(value)
    if not np.isfinite(result) or (result < 0 if allow_zero else result <= 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _non_negative(value: Any, name: str) -> float:
    return _positive(value, name, allow_zero=True)
