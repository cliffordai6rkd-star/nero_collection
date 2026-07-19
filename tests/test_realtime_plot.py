from __future__ import annotations

from collections import deque
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest
import yaml

from nero_collection.config import (
    DynamicsProcessingConfig,
    InverseDynamicsConfig,
    RealtimePlotConfig,
    StateParamConfig,
    _parse_realtime_plot,
)
from nero_collection.contact_wrench import PinocchioJointTorqueResidualEstimator, solve_damped_wrench
from nero_collection.realtime_plot import (
    RealtimeJointPlotter,
    SlidingJointBuffer,
    _RealtimeSample,
    _reconstruct_realtime_sample,
)


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
    assert config.inverse_dynamics.manifest_path is None


def test_realtime_plot_config_resolves_identified_manifest(tmp_path: Path) -> None:
    config = _parse_realtime_plot(
        {"inverse_dynamics": {"manifest_path": "results/dynamics_manifest.yaml"}},
        tmp_path,
    )

    assert config.inverse_dynamics.manifest_path == (
        tmp_path / "results" / "dynamics_manifest.yaml"
    ).resolve()


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
    assert estimate.tau_model == pytest.approx(tau_id)
    assert estimate.tau_friction == pytest.approx(np.zeros(7))
    assert estimate.tau_bias == pytest.approx(np.zeros(7))
    assert estimate.tau_residual == pytest.approx(expected_residual)


@pytest.mark.skipif(find_spec("pinocchio") is None, reason="Pinocchio is not installed")
def test_inverse_dynamics_estimator_applies_identified_friction_and_bias(
    tmp_path: Path,
) -> None:
    import pinocchio as pin

    urdf_path = (
        Path(__file__).resolve().parents[1]
        / "urdf"
        / "nero"
        / "nero_with_gripper.urdf"
    )
    coulomb = np.linspace(0.1, 0.7, 7)
    viscous = np.linspace(0.01, 0.07, 7)
    bias = np.linspace(-0.3, 0.3, 7)
    velocity_scale = 0.02
    manifest_path = tmp_path / "dynamics_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "identified_urdf": str(urdf_path),
                "joint_names": [f"joint{index}" for index in range(1, 8)],
                "friction": {
                    "coulomb_nm": coulomb.tolist(),
                    "viscous_nm_per_rad_s": viscous.tolist(),
                    "coulomb_velocity_scale_rad_s": velocity_scale,
                },
                "joint_torque_bias_nm": bias.tolist(),
            }
        ),
        encoding="utf-8",
    )
    estimator = PinocchioJointTorqueResidualEstimator(
        InverseDynamicsConfig(urdf_path=urdf_path, manifest_path=manifest_path)
    )
    q = pin.neutral(estimator.model)
    dq = np.linspace(-0.3, 0.3, 7)
    ddq = np.linspace(0.2, -0.2, 7)
    tau_id = np.asarray(pin.rnea(estimator.model, estimator.data, q, dq, ddq)).copy()
    expected_friction = coulomb * np.tanh(dq / velocity_scale) + viscous * dq
    expected_model = tau_id + expected_friction + bias
    expected_residual = np.linspace(-0.2, 0.2, 7)

    estimate = estimator.estimate(q, dq, ddq, expected_model - expected_residual)

    assert estimate.tau_id == pytest.approx(tau_id)
    assert estimate.tau_friction == pytest.approx(expected_friction)
    assert estimate.tau_bias == pytest.approx(bias)
    assert estimate.tau_model == pytest.approx(expected_model)
    assert estimate.tau_residual == pytest.approx(expected_residual)


@pytest.mark.skipif(find_spec("pinocchio") is None, reason="Pinocchio is not installed")
def test_inverse_dynamics_estimator_rejects_manifest_for_another_urdf(
    tmp_path: Path,
) -> None:
    urdf_path = (
        Path(__file__).resolve().parents[1]
        / "urdf"
        / "nero"
        / "nero_with_gripper.urdf"
    )
    manifest_path = tmp_path / "dynamics_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "identified_urdf": str(tmp_path / "another.urdf"),
                "joint_names": [f"joint{index}" for index in range(1, 8)],
                "friction": {
                    "coulomb_nm": [0.0] * 7,
                    "viscous_nm_per_rad_s": [0.0] * 7,
                    "coulomb_velocity_scale_rad_s": 0.02,
                },
                "joint_torque_bias_nm": [0.0] * 7,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="manifest/URDF mismatch"):
        PinocchioJointTorqueResidualEstimator(
            InverseDynamicsConfig(urdf_path=urdf_path, manifest_path=manifest_path)
        )


def test_fixed_lag_realtime_reconstruction_uses_q_instead_of_firmware_velocity() -> None:
    timestamps = np.arange(101, dtype=np.int64) * 10_000 + 1_000_000
    time_s = (timestamps - timestamps[0]) * 1e-6
    phases = np.linspace(0.0, 0.6, 7)
    samples = []
    for timestamp_us, time_value in zip(timestamps, time_s):
        q = np.sin(time_value + phases)
        samples.append(
            _RealtimeSample(
                timestamp_us=int(timestamp_us),
                q=q,
                q_timestamp_us=int(timestamp_us),
                q_acquired_timestamp_us=int(timestamp_us),
                motor_timestamp_us=np.full(7, timestamp_us, dtype=np.int64),
                motor_acquired_timestamp_us=np.full(7, timestamp_us, dtype=np.int64),
                dq_firmware=np.zeros(7),
                ddq_adapter=np.zeros(7),
                tau=np.full(7, 0.2),
            )
        )
    target = samples[50]
    processing = DynamicsProcessingConfig(
        enabled=True,
        spline_smoothing_rad2=0.0,
        torque_lowpass_hz=12.0,
        torque_median_window=3,
        min_samples=20,
    )

    q, dq, ddq, tau = _reconstruct_realtime_sample(
        deque(samples),
        target,
        delay_us=500_000,
        processing=processing,
    )

    target_time = time_s[50]
    assert q == pytest.approx(np.sin(target_time + phases), abs=1e-5)
    assert dq == pytest.approx(np.cos(target_time + phases), abs=1e-4)
    assert ddq == pytest.approx(-np.sin(target_time + phases), abs=2e-3)
    assert tau == pytest.approx(np.full(7, 0.2), abs=1e-6)


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
