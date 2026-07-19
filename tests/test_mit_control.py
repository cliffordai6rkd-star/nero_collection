from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from nero_collection.arms.pyagx import PyAgxArmAdapter
from nero_collection.config import ArmEndpointConfig, _parse_command, load_config


class FakeRobot:
    def __init__(self) -> None:
        self.calls: list[dict[str, float | int]] = []

    def move_mit(self, **kwargs):
        self.calls.append(kwargs)
        return True


class StatusRobot:
    def __init__(self, ctrl_mode: int) -> None:
        self.status = SimpleNamespace(msg=SimpleNamespace(ctrl_mode=ctrl_mode))

    def get_arm_status(self):
        return self.status


class LeaderFeedbackRobot(StatusRobot):
    def __init__(self) -> None:
        super().__init__(0x01)
        self.leader_timestamp = 1.0
        self.mode_calls: list[str] = []

    def get_leader_joint_angles(self):
        return SimpleNamespace(msg=[0.0] * 7, timestamp=self.leader_timestamp)

    def set_normal_mode(self) -> None:
        self.mode_calls.append("normal")

    def set_leader_mode(self) -> None:
        self.mode_calls.append("leader")
        self.leader_timestamp = 2.0


class FakeGripper:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float, float]] = []
        self.disable_calls = 0
        self.enabled = True

    def move_gripper_m(self, *, value: float, force: float) -> None:
        self.calls.append(("width", value, force))

    def move_gripper_deg(self, *, value: float, force: float) -> None:
        self.calls.append(("angle", value, force))

    def disable_gripper(self) -> bool:
        self.disable_calls += 1
        self.enabled = False
        return True

    def get_gripper_status(self):
        return SimpleNamespace(
            timestamp=123.0,
            msg=SimpleNamespace(
                value=25.0,
                force=0.2,
                mode="angle",
                foc_status=SimpleNamespace(driver_enable_status=self.enabled),
            )
        )

    def get_gripper_ctrl_states(self):
        return SimpleNamespace(
            timestamp=124.0,
            msg=SimpleNamespace(value=0.04, force=0.0, status_code=1),
        )


def test_master_slave_config_has_valid_control_parameters() -> None:
    config = load_config("configs/master_slave_can.yaml")

    command = config.teleop.command
    assert command.control_mode in {"mit", "position"}
    assert len(command.mit.kp) == 7
    assert len(command.mit.kd) == 7
    assert all(0.0 <= value <= 500.0 for value in command.mit.kp)
    assert all(-5.0 <= value <= 5.0 for value in command.mit.kd)
    assert config.gripper.teleop_enabled is True
    assert config.gripper.attach_to == "both"
    assert config.realtime_plot.inverse_dynamics.manifest_path is not None
    assert config.realtime_plot.inverse_dynamics.manifest_path.is_file()
    assert config.dynamics_processing.enabled is True
    assert config.dynamics_processing.state_method == "spline"


@pytest.mark.parametrize(
    "mit",
    [
        {"kp": [1.0] * 6},
        {"kp": [501.0] * 7},
        {"kd": [5.1] * 7},
        {"v_des": [45.1] * 7},
        {"t_ff": [9.0] * 7},
    ],
)
def test_mit_config_rejects_invalid_vectors(mit: dict[str, list[float]]) -> None:
    with pytest.raises(ValueError):
        _parse_command({"control_mode": "mit", "mit": mit})


def test_pyagx_adapter_sends_all_seven_mit_commands() -> None:
    adapter = PyAgxArmAdapter(ArmEndpointConfig(name="follower"))
    robot = FakeRobot()
    adapter._robot = robot

    adapter.command_joint_impedance(
        q=np.arange(7, dtype=np.float64) * 0.1,
        v_des=np.zeros(7),
        kp=np.arange(1, 8, dtype=np.float64),
        kd=np.full(7, 0.8),
        t_ff=np.zeros(7),
    )

    assert [call["joint_index"] for call in robot.calls] == list(range(1, 8))
    assert robot.calls[3]["p_des"] == pytest.approx(0.3)
    assert robot.calls[6]["kp"] == pytest.approx(7.0)


def test_pyagx_adapter_rejects_sdk_without_move_mit() -> None:
    adapter = PyAgxArmAdapter(ArmEndpointConfig(name="follower"))
    adapter._robot = object()

    with pytest.raises(RuntimeError, match="does not expose Nero move_mit"):
        adapter.validate_joint_impedance_support()


@pytest.mark.parametrize(("ctrl_mode", "expected"), [(0x06, "leader"), (0x01, "follower")])
def test_pyagx_adapter_reads_control_role(ctrl_mode: int, expected: str) -> None:
    adapter = PyAgxArmAdapter(ArmEndpointConfig(name="arm"))
    adapter._robot = StatusRobot(ctrl_mode)

    assert adapter.read_control_role() == expected


def test_pyagx_adapter_refreshes_cached_control_role() -> None:
    adapter = PyAgxArmAdapter(ArmEndpointConfig(name="arm"))
    adapter._robot = StatusRobot(0x01)
    adapter._configured_role = "leader"

    assert adapter.read_control_role(refresh=True) == "follower"


def test_pyagx_adapter_verifies_commanded_leader_from_fresh_joint_feedback() -> None:
    adapter = PyAgxArmAdapter(ArmEndpointConfig(name="arm"))
    adapter._robot = LeaderFeedbackRobot()

    adapter.set_leader_mode()

    assert adapter._robot.mode_calls == ["leader", "leader", "leader"]
    assert adapter.read_control_role(refresh=True) == "leader"


def test_pyagx_adapter_commands_gripper_in_width_mode() -> None:
    adapter = PyAgxArmAdapter(ArmEndpointConfig(name="arm"))
    gripper = FakeGripper()
    adapter._gripper = gripper

    adapter.command_gripper(0.035, 2.0)

    assert gripper.calls == [("width", 0.035, 2.0)]


def test_pyagx_adapter_commands_gripper_in_angle_mode() -> None:
    adapter = PyAgxArmAdapter(ArmEndpointConfig(name="arm"))
    gripper = FakeGripper()
    adapter._gripper = gripper

    adapter.command_gripper(25.0, 2.0, mode="angle")

    assert gripper.calls == [("angle", 25.0, 2.0)]


def test_pyagx_adapter_reads_angle_mode_and_disables_leader_gripper() -> None:
    adapter = PyAgxArmAdapter(ArmEndpointConfig(name="arm"))
    gripper = FakeGripper()
    adapter._gripper = gripper

    state = adapter.read_gripper_state()
    adapter.disable_gripper()

    assert state.value == pytest.approx(25.0)
    assert state.mode == "angle"
    assert state.timestamp_us == 123_000_000
    assert gripper.disable_calls == 1


def test_pyagx_adapter_reads_leader_gripper_control_frame() -> None:
    adapter = PyAgxArmAdapter(ArmEndpointConfig(name="arm"))
    adapter._gripper = FakeGripper()

    state = adapter.read_leader_gripper_state()

    assert state.value == pytest.approx(0.04)
    assert state.mode == "width"
    assert state.timestamp_us == 124_000_000


def test_pyagx_adapter_rejects_gripper_control_frame_from_before_leader_mode() -> None:
    adapter = PyAgxArmAdapter(ArmEndpointConfig(name="arm"))
    adapter._gripper = FakeGripper()
    adapter._leader_mode_commanded = True
    adapter._leader_gripper_feedback_baseline = 124.0

    state = adapter.read_leader_gripper_state()

    assert np.isnan(state.value)
    assert state.mode == "unknown"
