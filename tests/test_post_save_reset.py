from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import nero_collection.cli as cli
import nero_collection.teleop.master_slave as master_slave_module
from nero_collection.arms.base import ArmState, GripperState
from nero_collection.config import (
    CollectionConfig,
    CommandConfig,
    GripperConfig,
    MitControlConfig,
    OutputConfig,
    TeleopConfig,
)
from nero_collection.teleop.master_slave import ArmPairRuntime, MasterSlaveTeleop


def test_reset_behavior_is_enabled_by_default() -> None:
    command = CommandConfig()

    assert command.reset_after_episode is True
    assert command.reset_test_sample_time == 5


class FakeLeader:
    name = "leader"

    def __init__(self, positions: list[np.ndarray] | None = None) -> None:
        self._positions = positions or [np.zeros(7, dtype=np.float64)]
        self.read_count = 0
        self.q = np.zeros(7, dtype=np.float64)
        self.role = "leader"
        self.move_commands: list[np.ndarray] = []
        self.gripper_value = 0.0
        self.gripper_mode = "width"

    def read_control_role(self, refresh: bool = False) -> str:
        return self.role

    def set_leader_mode(self) -> None:
        self.role = "leader"

    def set_follower_mode(self) -> None:
        self.role = "follower"

    def enable(self) -> None:
        pass

    def move_joints(self, q: np.ndarray) -> None:
        target = np.asarray(q, dtype=np.float64).copy()
        self.move_commands.append(target)
        self.q = target

    def wait_motion_done(self, timeout_s: float, poll_interval_s: float = 0.1) -> bool:
        return True

    def read_state(self) -> ArmState:
        zeros = np.zeros_like(self.q)
        return ArmState(
            q=self.q.copy(),
            dq=zeros.copy(),
            ddq=zeros.copy(),
            ee_pose=np.eye(4, dtype=np.float64),
            torque=zeros.copy(),
            current=zeros.copy(),
            timestamp_us=self.read_count,
        )

    def read_leader_joint_positions(self) -> np.ndarray:
        index = min(self.read_count, len(self._positions) - 1)
        self.read_count += 1
        return self._positions[index].copy()

    def read_gripper_state(self) -> GripperState:
        return GripperState(value=self.gripper_value, force=0.0, timestamp_us=0, mode=self.gripper_mode)


class BiasedFollower:
    name = "follower"

    def __init__(self, bias: np.ndarray) -> None:
        self.bias = np.asarray(bias, dtype=np.float64)
        self.q = np.zeros_like(self.bias)
        self.move_commands: list[np.ndarray] = []
        self.command_joint_calls: list[np.ndarray] = []
        self.command_mit_calls: list[tuple[np.ndarray, ...]] = []
        self.read_count = 0
        self.read_counts_at_move: list[int] = []
        self.role = "follower"
        self.gripper_value = 0.0
        self.gripper_mode = "width"
        self.gripper_commands: list[tuple[float, float, str]] = []

    def read_control_role(self, refresh: bool = False) -> str:
        return self.role

    def set_leader_mode(self) -> None:
        self.role = "leader"

    def set_follower_mode(self) -> None:
        self.role = "follower"

    def enable(self) -> None:
        pass

    def move_joints(self, q: np.ndarray) -> None:
        target = np.asarray(q, dtype=np.float64).copy()
        self.move_commands.append(target)
        self.read_counts_at_move.append(self.read_count)
        self.q = target + self.bias

    def wait_motion_done(self, timeout_s: float, poll_interval_s: float = 0.1) -> bool:
        return True

    def command_joint_positions(self, q: np.ndarray) -> None:
        self.command_joint_calls.append(np.asarray(q, dtype=np.float64).copy())

    def command_joint_impedance(
        self,
        q: np.ndarray,
        v_des: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        t_ff: np.ndarray,
    ) -> None:
        self.command_mit_calls.append(
            tuple(np.asarray(value, dtype=np.float64).copy() for value in (q, v_des, kp, kd, t_ff))
        )

    def read_state(self) -> ArmState:
        self.read_count += 1
        zeros = np.zeros_like(self.q)
        return ArmState(
            q=self.q.copy(),
            dq=zeros.copy(),
            ddq=zeros.copy(),
            ee_pose=np.eye(4, dtype=np.float64),
            torque=zeros.copy(),
            current=zeros.copy(),
            timestamp_us=self.read_count,
        )

    def read_gripper_state(self) -> GripperState:
        return GripperState(value=self.gripper_value, force=0.0, timestamp_us=0, mode=self.gripper_mode)

    def command_gripper(self, value: float, force_n: float, mode: str = "width") -> None:
        self.gripper_commands.append((float(value), float(force_n), mode))
        self.gripper_value = float(value)
        self.gripper_mode = mode


def _teleop_with_fakes(
    command: CommandConfig,
    leader: FakeLeader,
    follower: BiasedFollower,
) -> MasterSlaveTeleop:
    config = CollectionConfig(
        teleop=TeleopConfig(command=command),
        output=OutputConfig(directory=Path(".")),
        gripper=GripperConfig(enabled=False),
    )
    teleop = MasterSlaveTeleop.__new__(MasterSlaveTeleop)
    teleop.config = config
    teleop.pairs = (
        ArmPairRuntime(
            name="main",
            leader=leader,
            follower=follower,
            rest_q_leader=np.zeros(7, dtype=np.float64),
            rest_q_follower=np.zeros(7, dtype=np.float64),
        ),
    )
    teleop.arm_names = ("main",)
    teleop._teleop_reference = {}
    teleop._hold_after_reset = False
    teleop._parked = False
    teleop._unrecorded_teleop = False
    teleop._last_gripper_command = {}
    teleop._last_gripper_command_mode = {}
    teleop._last_gripper_command_t = {}
    teleop._gripper_command_announced = set()
    teleop._gripper_feedback_warned = set()
    teleop._leader_gripper_feedback_timestamp_us = {}
    teleop._leader_gripper_feedback_change_t = {}
    teleop._leader_gripper_stale_warned = set()
    return teleop


def test_reset_uses_five_sample_mean_to_fine_tune(monkeypatch: pytest.MonkeyPatch) -> None:
    bias = np.array([0.03, -0.02, 0.01, 0.0, -0.01, 0.02, -0.03])
    leader = FakeLeader()
    follower = BiasedFollower(bias)
    command = CommandConfig(
        idle_rate_hz=100.0,
        reset_timeout_s=2.0,
        reset_wait_s=0.0,
        reset_test_sample_time=5,
        reset_error_limit_rad=0.001,
        joint_step_limit_rad=0.08,
        idle_follow_enabled=True,
        reset_interpolation_enabled=False,
    )
    teleop = _teleop_with_fakes(command, leader, follower)
    monkeypatch.setattr(master_slave_module.time, "sleep", lambda _seconds: None)

    teleop.reset_to_rest()

    assert follower.read_counts_at_move == [1, 7]
    assert follower.read_count == 12
    assert leader.role == "follower"
    assert follower.role == "follower"
    assert len(leader.move_commands) == 2
    assert all(np.allclose(target, 0.0) for target in leader.move_commands)
    assert np.allclose(follower.move_commands[0], np.zeros(7))
    assert np.allclose(follower.move_commands[1], -bias)
    assert teleop._hold_after_reset is True
    assert teleop._parked is True
    assert teleop._teleop_reference == {}

    teleop.idle_step()
    assert follower.command_joint_calls == []


def test_next_teleop_waits_until_leader_is_near_follower(monkeypatch: pytest.MonkeyPatch) -> None:
    leader = FakeLeader(
        [
            np.full(7, 0.30, dtype=np.float64),
            np.full(7, 0.15, dtype=np.float64),
            np.full(7, 0.05, dtype=np.float64),
        ]
    )
    follower = BiasedFollower(np.zeros(7, dtype=np.float64))
    command = CommandConfig(
        pre_teleop_align_enabled=True,
        pre_teleop_align_error_limit_rad=0.10,
        idle_follow_enabled=True,
    )
    teleop = _teleop_with_fakes(command, leader, follower)
    teleop._hold_after_reset = True
    leader.role = "follower"
    monkeypatch.setattr(master_slave_module.time, "sleep", lambda _seconds: None)

    teleop.enter_teleop()

    assert leader.role == "leader"
    assert leader.read_count == 3
    assert teleop._hold_after_reset is False
    leader_q0, follower_q0 = teleop._teleop_reference["main"]
    assert np.allclose(leader_q0, 0.05)
    assert np.allclose(follower_q0, 0.0)


def test_mit_control_dispatches_configured_joint_parameters() -> None:
    leader = FakeLeader([np.full(7, 0.04, dtype=np.float64)])
    follower = BiasedFollower(np.zeros(7, dtype=np.float64))
    mit = MitControlConfig(
        kp=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
        kd=(0.1,) * 7,
        v_des=(0.2,) * 7,
        t_ff=(0.3,) * 7,
    )
    command = CommandConfig(
        control_mode="mit",
        mit=mit,
        teleop_mapping="absolute",
        joint_step_limit_rad=0.08,
    )
    teleop = _teleop_with_fakes(command, leader, follower)

    teleop._command_follower(teleop.pairs[0], np.full(7, 0.04, dtype=np.float64))

    assert follower.command_joint_calls == []
    assert len(follower.command_mit_calls) == 1
    q, v_des, kp, kd, t_ff = follower.command_mit_calls[0]
    assert np.allclose(q, 0.04)
    assert np.allclose(v_des, mit.v_des)
    assert np.allclose(kp, mit.kp)
    assert np.allclose(kd, mit.kd)
    assert np.allclose(t_ff, mit.t_ff)


def test_t_mode_enters_unrecorded_master_slave_teleop() -> None:
    leader = FakeLeader(
        [
            np.full(7, 0.04, dtype=np.float64),
            np.full(7, 0.06, dtype=np.float64),
        ]
    )
    leader.role = "follower"
    follower = BiasedFollower(np.zeros(7, dtype=np.float64))
    teleop = _teleop_with_fakes(
        CommandConfig(pre_teleop_align_enabled=False, teleop_mapping="relative_offset"),
        leader,
        follower,
    )

    teleop.enter_unrecorded_teleop()
    teleop.idle_step()

    assert leader.role == "leader"
    assert follower.role == "follower"
    assert teleop._unrecorded_teleop is True
    assert teleop._parked is False
    assert "main" in teleop._teleop_reference
    assert len(follower.command_joint_calls) == 1
    assert np.allclose(follower.command_joint_calls[0], 0.02)


def test_gripper_teleop_maps_leader_width_to_follower() -> None:
    leader = FakeLeader()
    leader.gripper_value = 0.04
    follower = BiasedFollower(np.zeros(7, dtype=np.float64))
    follower.gripper_value = 0.01
    teleop = _teleop_with_fakes(CommandConfig(), leader, follower)
    teleop.config = CollectionConfig(
        teleop=teleop.config.teleop,
        output=OutputConfig(directory=Path(".")),
        gripper=GripperConfig(
            enabled=True,
            attach_to="both",
            teleop_enabled=True,
            scale=1.5,
            offset_m=0.005,
            min_width_m=0.0,
            max_width_m=0.06,
            force_n=2.0,
            command_rate_hz=30.0,
            deadband_m=0.0005,
        ),
    )
    teleop._last_gripper_command = {}
    teleop._last_gripper_command_mode = {}
    teleop._last_gripper_command_t = {}
    values: dict[str, tuple[str, np.ndarray]] = {}

    teleop._update_gripper_teleop(values)

    assert follower.gripper_commands == [(0.06, 2.0, "width")]
    assert values["gripper_leader"][1] == pytest.approx([0.04])
    assert values["gripper_follower"][1] == pytest.approx([0.01])
    assert values["gripper_cmd"][1] == pytest.approx([0.06])


def test_gripper_teleop_ignores_non_width_input() -> None:
    leader = FakeLeader()
    leader.gripper_value = 25.0
    leader.gripper_mode = "angle"
    follower = BiasedFollower(np.zeros(7, dtype=np.float64))
    teleop = _teleop_with_fakes(CommandConfig(), leader, follower)
    teleop.config = CollectionConfig(
        teleop=teleop.config.teleop,
        output=OutputConfig(directory=Path(".")),
        gripper=GripperConfig(
            enabled=True,
            attach_to="both",
            teleop_enabled=True,
        ),
    )
    values: dict[str, tuple[str, np.ndarray]] = {}

    teleop._update_gripper_teleop(values)

    assert follower.gripper_commands == []
    assert values["gripper_cmd"][1] == pytest.approx([np.nan], nan_ok=True)
    assert "gripper_force" not in values
    assert "gripper_mode_leader" not in values


def test_wait_for_record_start_handles_t_then_r(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class Keys:
        def __init__(self) -> None:
            self.keys = iter(("t", "r"))

        def read_key(self, _timeout: float) -> str:
            return next(self.keys)

    class Teleop:
        def enter_unrecorded_teleop(self) -> None:
            events.append("unrecorded_teleop")

        def idle_step(self) -> None:
            events.append("idle")

    config = CollectionConfig(
        teleop=TeleopConfig(command=CommandConfig(idle_rate_hz=100.0)),
        output=OutputConfig(directory=Path(".")),
    )
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    cli._wait_for_record_start(Teleop(), Keys(), config, None)

    assert events == ["unrecorded_teleop"]


@pytest.mark.parametrize(
    ("save", "expected_tail"),
    [
        (True, ["save", "reset"]),
        (False, ["reset"]),
    ],
)
def test_collection_resets_only_after_successful_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    save: bool,
    expected_tail: list[str],
) -> None:
    events: list[str] = []

    class FakeTeleop:
        arm_names = ("main",)

        def __init__(self, _config: CollectionConfig) -> None:
            pass

        def start(self) -> None:
            pass

        def shutdown(self) -> None:
            pass

        def enter_teleop(self) -> None:
            pass

        def enter_idle_follow(self) -> None:
            pass

        def reset_to_rest(self) -> None:
            events.append("reset")

    class FakeCameras:
        @classmethod
        def from_config(cls, _config: object) -> "FakeCameras":
            return cls()

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    class FakeKeys:
        is_tty = True

        def __enter__(self) -> "FakeKeys":
            return self

        def __exit__(self, *_args: object) -> None:
            pass

    class FakeBuffer:
        sample_count = 0

        def __init__(self, **_kwargs: object) -> None:
            pass

        def save(self, _path: Path) -> None:
            events.append("save")

    monkeypatch.setattr(cli, "MasterSlaveTeleop", FakeTeleop)
    monkeypatch.setattr(cli, "CameraManager", FakeCameras)
    monkeypatch.setattr(cli, "TerminalKeys", FakeKeys)
    monkeypatch.setattr(cli, "EpisodeBuffer", FakeBuffer)
    monkeypatch.setattr(cli, "_wait_for_record_start", lambda *_args: None)
    monkeypatch.setattr(cli, "_record_episode", lambda *_args: None)
    monkeypatch.setattr(cli, "_wait_for_save_choice", lambda *_args: save)

    config = CollectionConfig(
        teleop=TeleopConfig(command=CommandConfig(reset_after_episode=True)),
        output=OutputConfig(directory=tmp_path),
    )

    assert cli.run_collection(config, episode_limit=1) == 0
    assert events == expected_tail
