from pathlib import Path

import numpy as np
import pytest

from nero_collection.config import (
    CollectionConfig,
    DynamicsProcessingConfig,
    OutputConfig,
    StateParamConfig,
    TeleopConfig,
    _parse_state_param,
)
from nero_collection.filters import LowPassVelocityDifferentiator, OnePoleLowPass
from nero_collection.h5_writer import EpisodeBuffer
from nero_collection.dynamics_processing import resample_columns


def test_lowpass_velocity_differentiator_smooths_velocity_step() -> None:
    estimator = LowPassVelocityDifferentiator(cutoff_hz=10.0)

    first = estimator.apply(np.array([0.0]), timestamp_us=1_000_000)
    second = estimator.apply(np.array([1.0]), timestamp_us=1_010_000)
    third = estimator.apply(np.array([1.0]), timestamp_us=1_020_000)

    assert np.allclose(first, 0.0)
    assert 0.0 < second[0] < 100.0
    assert 0.0 < third[0] < second[0]


def test_lowpass_velocity_differentiator_rejects_nonincreasing_time() -> None:
    estimator = LowPassVelocityDifferentiator(cutoff_hz=10.0)
    estimator.apply(np.array([0.0]), timestamp_us=1_000_000)

    with pytest.raises(ValueError, match="strictly increasing"):
        estimator.apply(np.array([1.0]), timestamp_us=1_000_000)


def test_causal_median_rejects_single_sample_spike_before_iir() -> None:
    filt = OnePoleLowPass(cutoff_hz=10.0, median_window=3)

    first = filt.apply(np.array([1.0]), timestamp_us=1_000_000)
    spike = filt.apply(np.array([18.0]), timestamp_us=1_010_000)
    recovered = filt.apply(np.array([1.1]), timestamp_us=1_020_000)

    assert first == pytest.approx([1.0])
    assert spike == pytest.approx([1.0])
    assert 1.0 < recovered[0] < 1.1


def test_iir_alpha_uses_actual_timestamp_interval() -> None:
    filt = OnePoleLowPass(cutoff_hz=10.0, median_window=1)
    filt.apply(np.array([0.0]), timestamp_us=1_000_000)

    result = filt.apply(np.array([1.0]), timestamp_us=1_010_000)

    expected_alpha = 1.0 - np.exp(-2.0 * np.pi * 10.0 * 0.01)
    assert result == pytest.approx([expected_alpha])


@pytest.mark.parametrize("window", [0, 2, 4])
def test_state_config_rejects_invalid_median_window(window: int) -> None:
    with pytest.raises(ValueError, match="positive odd"):
        _parse_state_param({"median_window": window})


def test_episode_buffer_replaces_adapter_acceleration_with_filtered_derivative() -> None:
    config = CollectionConfig(
        teleop=TeleopConfig(),
        output=OutputConfig(directory=Path(".")),
        robot_states={
            "velocity": StateParamConfig(enabled=True, lowpass=False),
            "acceleration": StateParamConfig(
                enabled=True,
                lowpass=False,
                velocity_lowpass_cutoff_hz=10.0,
            ),
        },
    )
    buffer = EpisodeBuffer(config=config, arm_names=("main",), sample_rate_hz=100.0)

    buffer.append_teleop(
        1_000_000,
        {
            "dq_follower": ("velocity", np.array([0.0])),
            "ddq_follower": ("acceleration", np.array([999.0])),
        },
    )
    processed = buffer.append_teleop(
        1_010_000,
        {
            "dq_follower": ("velocity", np.array([1.0])),
            "ddq_follower": ("acceleration", np.array([999.0])),
        },
    )

    assert np.allclose(buffer.teleop_data["dq_follower"], [[0.0], [1.0]])
    assert np.allclose(buffer.teleop_data["ddq_follower"][0], [0.0])
    assert 0.0 < buffer.teleop_data["ddq_follower"][1][0] < 100.0
    assert processed["dq_follower"][1] == pytest.approx([1.0])
    assert processed["ddq_follower"][1] == pytest.approx(
        buffer.teleop_data["ddq_follower"][1]
    )


def test_episode_save_reconstructs_dynamics_from_q_and_preserves_raw_fields(
    tmp_path: Path,
) -> None:
    h5py = pytest.importorskip("h5py")
    config = CollectionConfig(
        teleop=TeleopConfig(),
        output=OutputConfig(directory=tmp_path),
        dynamics_processing=DynamicsProcessingConfig(
            enabled=True,
            state_method="spline",
            spline_smoothing_rad2=0.0,
            torque_lowpass_hz=12.0,
            torque_median_window=3,
            min_samples=20,
        ),
        robot_states={
            "q": StateParamConfig(enabled=True),
            "velocity": StateParamConfig(enabled=True),
            "acceleration": StateParamConfig(enabled=True),
            "torque": StateParamConfig(enabled=True),
        },
    )
    buffer = EpisodeBuffer(config=config, arm_names=("main",), sample_rate_hz=100.0)
    time_s = np.arange(100, dtype=np.float64) * 0.01
    timestamps = (1_000_000 + time_s * 1e6).astype(np.int64)
    phases = np.linspace(0.0, 0.6, 7)
    q = np.sin(time_s[:, None] + phases[None, :])
    tau = 0.2 * np.cos(time_s[:, None] + phases[None, :])
    tau[50, 3] += 5.0
    for index, timestamp_us in enumerate(timestamps):
        buffer.append_teleop(
            int(timestamp_us),
            {
                "q_follower": ("q", q[index]),
                "q_timestamp_follower_us": ("timestamp", np.asarray([timestamp_us])),
                "q_acquired_timestamp_follower_us": (
                    "timestamp",
                    np.asarray([timestamp_us]),
                ),
                "dq_follower": ("velocity", np.zeros(7)),
                "ddq_follower": ("acceleration", np.full(7, 999.0)),
                "tau_follower": ("torque", tau[index]),
                "motor_timestamp_follower_us": (
                    "timestamp",
                    np.full(7, timestamp_us, dtype=np.int64),
                ),
                "motor_acquired_timestamp_follower_us": (
                    "timestamp",
                    np.full(7, timestamp_us, dtype=np.int64),
                ),
            },
        )

    output = buffer.save(tmp_path / "episode.h5")

    with h5py.File(output, "r") as h5:
        teleop = h5["teleop"]
        assert h5.attrs["format"] == "factr_multimodal_episode/v3"
        assert np.max(np.abs(teleop["dq_follower"][:])) > 0.5
        assert np.max(np.abs(teleop["ddq_follower"][:])) > 0.5
        assert np.allclose(teleop["dq_follower_firmware_raw"][:], 0.0)
        assert np.allclose(teleop["ddq_follower_adapter_raw"][:], 999.0)
        assert np.allclose(teleop["q_follower_raw"][:], q)
        assert np.allclose(teleop["tau_follower_raw"][:], tau)
        assert abs(teleop["tau_follower"][50, 3]) < abs(tau[50, 3])
        assert teleop["dq_follower"].attrs["timestamp_path"] == "teleop/timestamp_us"
        assert bool(teleop["tau_follower"].attrs["zero_phase"]) is True

def test_resampling_falls_back_when_sdk_motor_timestamp_is_stale() -> None:
    timeline = np.arange(20, dtype=np.int64) * 10_000 + 1_000_000
    stale_motor_timestamp = np.full((20, 2), 900_000, dtype=np.int64)
    values = np.column_stack((np.arange(20), np.arange(20) * 2)).astype(np.float64)

    result = resample_columns(
        stale_motor_timestamp,
        values,
        timeline,
        fallback_source_timestamp_us=timeline,
    )

    assert result == pytest.approx(values)
