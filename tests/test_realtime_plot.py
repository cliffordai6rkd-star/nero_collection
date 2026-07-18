from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

from nero_collection.config import (
    InverseDynamicsConfig,
    RealtimePlotConfig,
    StateParamConfig,
    _parse_realtime_plot,
)
from nero_collection.contact_wrench import PinocchioJointTorqueResidualEstimator, solve_damped_wrench
from nero_collection.realtime_plot import RealtimeJointPlotter, SlidingJointBuffer


def test_realtime_plot_config_defaults_to_ten_second_window() -> None:
    config = _parse_realtime_plot({})

    assert config.enabled is False
    assert config.window_s == pytest.approx(10.0)
    assert config.update_rate_hz == pytest.approx(20.0)
    assert config.inverse_dynamics.delay_s == pytest.approx(0.5)
    assert config.inverse_dynamics.locked_joint_names == (
        "gripper",
        "gripper_joint1",
        "gripper_joint2",
    )


@pytest.mark.parametrize(
    "data",
    [
        {"window_s": 0.0},
        {"window_s": float("nan")},
        {"update_rate_hz": 0.0},
        {"inverse_dynamics": {"delay_s": -0.1}},
        {"inverse_dynamics": {"locked_joint_names": ["joint7", ""]}},
        {"inverse_dynamics": {"gravity_m_s2": [0.0, float("nan"), -9.81]}},
    ],
)
def test_realtime_plot_config_rejects_invalid_rates(data: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        _parse_realtime_plot(data)


def test_sliding_joint_buffer_keeps_only_latest_ten_seconds() -> None:
    buffer = SlidingJointBuffer(window_s=10.0)
    for timestamp_s, value in ((0, 1.0), (5, 2.0), (11, 3.0)):
        joints = np.full(7, value, dtype=np.float64)
        tau_ext = np.full(7, value * 3, dtype=np.float64)
        buffer.append(timestamp_s * 1_000_000, joints, joints * 2, tau_ext)

    time_s, q, tau, tau_ext = buffer.arrays()

    assert time_s == pytest.approx([-6.0, 0.0])
    assert q.shape == (2, 7)
    assert np.allclose(q[:, 0], [2.0, 3.0])
    assert np.allclose(tau[:, 0], [4.0, 6.0])
    assert tau_ext.shape == (2, 7)
    assert np.allclose(tau_ext[:, 0], [6.0, 9.0])


def test_sliding_joint_buffer_rejects_non_seven_dimensional_data() -> None:
    buffer = SlidingJointBuffer(window_s=10.0)

    with pytest.raises(RuntimeError, match="7D q"):
        buffer.append(1, np.zeros(6), np.zeros(7), np.zeros(7))

    with pytest.raises(RuntimeError, match="7D tau_ext"):
        buffer.append(1, np.zeros(7), np.zeros(7), np.zeros(6))


def test_damped_wrench_maps_joint_residual_and_reports_nullspace_error() -> None:
    jacobian = np.zeros((6, 7), dtype=np.float64)
    jacobian[:, :6] = np.eye(6)
    tau_residual = np.arange(1, 8, dtype=np.float64)

    wrench, error, condition = solve_damped_wrench(jacobian, tau_residual, damping=1e-6)

    assert wrench == pytest.approx(tau_residual[:6], rel=1e-9)
    assert error == pytest.approx(7.0 / np.linalg.norm(tau_residual), rel=1e-9)
    assert condition == pytest.approx(1.0)


@pytest.mark.skipif(find_spec("pinocchio") is None, reason="Pinocchio is not installed")
def test_inverse_dynamics_estimator_returns_tau_id_minus_measured() -> None:
    import pinocchio as pin

    config = InverseDynamicsConfig(
        urdf_path=(
            Path(__file__).resolve().parents[1]
            / "urdf"
            / "nero"
            / "nero_with_gripper.urdf"
        )
    )
    estimator = PinocchioJointTorqueResidualEstimator(config)
    q = pin.neutral(estimator.model)
    dq = np.zeros(7, dtype=np.float64)
    ddq = np.zeros(7, dtype=np.float64)
    tau_id = np.asarray(pin.rnea(estimator.model, estimator.data, q, dq, ddq)).copy()
    expected_residual = np.linspace(-0.3, 0.3, 7)

    estimate = estimator.estimate(q, dq, ddq, tau_id - expected_residual)

    assert estimate.tau_id == pytest.approx(tau_id)
    assert estimate.tau_residual == pytest.approx(expected_residual)


@pytest.mark.skipif(find_spec("pinocchio") is None, reason="Pinocchio is not installed")
def test_realtime_plot_process_accepts_sample_and_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")
    enabled_state = StateParamConfig(enabled=True)
    plotter = RealtimeJointPlotter(
        RealtimePlotConfig(enabled=True, window_s=10.0, update_rate_hz=20.0),
        {
            "q": enabled_state,
            "velocity": enabled_state,
            "acceleration": enabled_state,
            "torque": enabled_state,
        },
    )
    values = {
        "q_follower": ("q", np.arange(7, dtype=np.float64)),
        "dq_follower": ("velocity", np.arange(7, dtype=np.float64) * 2.0),
        "ddq_follower": ("acceleration", np.arange(7, dtype=np.float64) * 0.5),
        "tau_follower": ("torque", np.arange(7, dtype=np.float64) * 3.0),
    }

    plotter.start()
    process = plotter._process
    assert process is not None
    plotter.append(1_000_000, values)
    plotter.close()

    assert process.exitcode == 0
