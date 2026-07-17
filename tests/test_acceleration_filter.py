from pathlib import Path

import numpy as np
import pytest

from nero_collection.config import CollectionConfig, OutputConfig, StateParamConfig, TeleopConfig
from nero_collection.filters import LowPassVelocityDifferentiator
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
    buffer.append_teleop(
        1_010_000,
        {
            "dq_follower": ("velocity", np.array([1.0])),
            "ddq_follower": ("acceleration", np.array([999.0])),
        },
    )

    assert np.allclose(buffer.teleop_data["dq_follower"], [[0.0], [1.0]])
    assert np.allclose(buffer.teleop_data["ddq_follower"][0], [0.0])
    assert 0.0 < buffer.teleop_data["ddq_follower"][1][0] < 100.0
