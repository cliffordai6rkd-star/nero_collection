from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

from calibration.common import StaticDataset, aggregate_static_poses, load_dataset, load_plan, save_dataset
from calibration.collect_static import _minimum_jerk_scale
from calibration.fit_static import main as fit_static_main
from calibration.validate_internal import main as validate_internal_main
from calibration.write_calibrated_urdf import main as write_urdf_main


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_minimum_jerk_scale_is_monotonic_with_stationary_endpoints() -> None:
    phase = np.linspace(0.0, 1.0, 1001)
    scale = np.asarray([_minimum_jerk_scale(value) for value in phase])

    assert scale[0] == pytest.approx(0.0)
    assert scale[-1] == pytest.approx(1.0)
    assert np.all(np.diff(scale) >= 0.0)
    assert (scale[1] - scale[0]) / (phase[1] - phase[0]) < 1e-4
    assert (scale[-1] - scale[-2]) / (phase[-1] - phase[-2]) < 1e-4


def test_static_dataset_round_trip_and_pose_aggregation(tmp_path: Path) -> None:
    q = np.arange(42, dtype=np.float64).reshape(6, 7) * 0.01
    tau = q * 2.0
    dataset = StaticDataset(
        q=q,
        tau=tau,
        current=np.full_like(q, np.nan),
        timestamp_us=np.arange(6, dtype=np.int64) + 100,
        pose_index=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32),
        round_index=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32),
        pose_names=("a", "b"),
        metadata={"label": "test"},
    )
    path = save_dataset(tmp_path / "samples.npz", dataset)

    loaded = load_dataset(path)
    names, q_pose, tau_pose, tau_std = aggregate_static_poses(loaded)

    assert names == ("a", "b")
    assert np.allclose(q_pose[0], q[1])
    assert np.allclose(q_pose[1], q[4])
    assert np.allclose(tau_pose, q_pose * 2.0)
    assert tau_std.shape == (2, 7)
    assert loaded.metadata == {"label": "test"}
    assert loaded.round_index.tolist() == [0, 0, 0, 1, 1, 1]


def test_terminal_static_fit_recovers_synthetic_parameters() -> None:
    pytest.importorskip("pinocchio")
    from calibration.static_model import TerminalStaticModel

    plan = load_plan(REPOSITORY_ROOT / "calibration" / "config.yaml")
    model = TerminalStaticModel(plan.model)
    q = np.stack([pose.q for pose in plan.poses])
    expected_parameters = model.nominal_terminal_parameters + np.asarray([0.08, 0.004, -0.003, 0.002])
    expected_bias = np.asarray([0.1, -0.2, 0.05, 0.03, -0.04, 0.02, -0.01])
    tau = model.predict(q, expected_parameters, expected_bias)

    result = model.fit(q, tau)

    assert result.regressor_rank == 4
    assert result.regressor_condition_number < 100.0
    assert np.allclose(result.terminal_parameters, expected_parameters, atol=1e-10)
    assert np.allclose(result.joint_bias_nm, expected_bias, atol=1e-10)
    assert result.overall_rmse_nm < 1e-10


def test_internal_validation_accepts_stable_synthetic_rounds(tmp_path: Path) -> None:
    pytest.importorskip("pinocchio")
    from calibration.static_model import TerminalStaticModel

    plan = load_plan(REPOSITORY_ROOT / "calibration" / "config.yaml")
    model = TerminalStaticModel(plan.model)
    q_pose = np.stack([pose.q for pose in plan.poses])
    terminal = model.nominal_terminal_parameters + np.asarray([0.08, 0.004, -0.003, 0.002])
    bias = np.asarray([0.1, -0.2, 0.05, 0.03, -0.04, 0.02, -0.01])
    tau_pose = model.predict(q_pose, terminal, bias)
    round_count = 3
    q = np.tile(q_pose, (round_count, 1))
    tau = np.tile(tau_pose, (round_count, 1))
    pose_index = np.tile(np.arange(len(plan.poses), dtype=np.int32), round_count)
    round_index = np.repeat(np.arange(round_count, dtype=np.int32), len(plan.poses))
    dataset = StaticDataset(
        q=q,
        tau=tau,
        current=np.full_like(q, np.nan),
        timestamp_us=np.arange(q.shape[0], dtype=np.int64) + 1,
        pose_index=pose_index,
        round_index=round_index,
        pose_names=tuple(pose.name for pose in plan.poses),
        metadata={"label": "synthetic_internal"},
    )
    data_path = save_dataset(tmp_path / "static.npz", dataset)
    fit_path = tmp_path / "fit.yaml"
    validation_path = tmp_path / "internal.yaml"

    assert fit_static_main(
        ["--config", str(plan.source_path), "--data", str(data_path), "--output", str(fit_path)]
    ) == 0
    assert validate_internal_main(
        [
            "--config",
            str(plan.source_path),
            "--fit",
            str(fit_path),
            "--data",
            str(data_path),
            "--output",
            str(validation_path),
        ]
    ) == 0

    validation = yaml.safe_load(validation_path.read_text(encoding="utf-8"))
    assert validation["accepted"] is True
    assert validation["validation_type"] == "internal_pose_holdout"
    assert validation["external_validation_performed"] is False
    assert validation["holdout_metrics"]["improvement_ratio"] > 0.99


def test_write_calibrated_urdf_collapses_terminal_inertias(tmp_path: Path) -> None:
    pytest.importorskip("pinocchio")
    source = REPOSITORY_ROOT / "urdf" / "nero" / "nero_with_gripper.urdf"
    copied_source = tmp_path / source.name
    copied_source.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    config = yaml.safe_load((REPOSITORY_ROOT / "calibration" / "config.yaml").read_text(encoding="utf-8"))
    config["collection_config"] = str(REPOSITORY_ROOT / "configs" / "master_slave_can.yaml")
    config["model"]["urdf_path"] = str(copied_source)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    plan = load_plan(config_path)
    from calibration.static_model import TerminalStaticModel

    model = TerminalStaticModel(plan.model)
    terminal_parameters = model.nominal_terminal_parameters + np.asarray([0.05, 0.002, -0.001, 0.0015])
    fit_path = tmp_path / "fit.yaml"
    fit_path.write_text(
        yaml.safe_dump(
            {
                "accepted": True,
                "fit": {
                    "terminal_parameters": terminal_parameters.tolist(),
                    "joint_torque_bias_nm": [0.0] * 7,
                },
            }
        ),
        encoding="utf-8",
    )
    validation_path = tmp_path / "validation.yaml"
    validation_path.write_text(
        yaml.safe_dump(
            {
                "accepted": True,
                "validation_type": "internal_pose_holdout",
                "external_validation_performed": False,
                "input": {"fit_report": str(fit_path)},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "nero_with_gripper_calibrated_static.urdf"
    manifest = tmp_path / "manifest.yaml"

    assert write_urdf_main(
        [
            "--config",
            str(config_path),
            "--fit",
            str(fit_path),
            "--validation",
            str(validation_path),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ]
    ) == 0

    root = ET.parse(output).getroot()
    link7 = next(link for link in root.findall("link") if link.get("name") == "link7")
    gripper_base = next(link for link in root.findall("link") if link.get("name") == "gripper_base")
    assert float(link7.find("inertial/mass").get("value")) == pytest.approx(terminal_parameters[0])
    assert gripper_base.find("inertial") is None
    manifest_data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    assert manifest_data["source_urdf"] == str(copied_source)
    assert np.allclose(manifest_data["joint_torque_bias_nm"], 0.0)
