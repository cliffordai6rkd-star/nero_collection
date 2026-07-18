from pathlib import Path

import numpy as np
import pytest

from nero_collection.config import (
    CollectionConfig,
    OutputConfig,
    StateParamConfig,
    TeleopConfig,
    _parse_state_param,
)
from nero_collection.filters import LowPassVelocityDifferentiator, OnePoleLowPass
from nero_collection.h5_writer import EpisodeBuffer


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
