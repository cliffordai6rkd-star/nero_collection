from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_TELEOP_MODES = {"master_slave", "meta_quest3_vr", "keyboard_3d_mouse"}


@dataclass(frozen=True)
class StateParamConfig:
    enabled: bool = True
    lowpass: bool = False
    lowpass_cutoff_hz: float = 12.0
    velocity_lowpass_cutoff_hz: float | None = None


@dataclass(frozen=True)
class OutputConfig:
    directory: Path
    prefix: str = "episode"


@dataclass(frozen=True)
class CameraConfig:
    name: str
    enabled: bool = True
    backend: str = "orbbec_dabai"
    serial_number: str | None = None
    width: int = 640
    height: int = 480
    fps: float = 30.0
    exposure: int | None = None
    depth: bool = False
    crop: tuple[int | None, int | None, int | None, int | None] = (0, None, 0, None)
    output_size: tuple[int, int] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArmEndpointConfig:
    name: str
    can_id: int | None = None
    channel: str = "can0"
    interface: str = "socketcan"
    bitrate: int = 1_000_000
    firmware: str = "V1_6"
    rest_q: tuple[float, ...] = ()
    config_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArmPairConfig:
    name: str
    leader: ArmEndpointConfig
    follower: ArmEndpointConfig


@dataclass(frozen=True)
class MitControlConfig:
    kp: tuple[float, ...] = (5.0,) * 7
    kd: tuple[float, ...] = (0.8,) * 7
    v_des: tuple[float, ...] = (0.0,) * 7
    t_ff: tuple[float, ...] = (0.0,) * 7


@dataclass(frozen=True)
class CommandConfig:
    sample_rate_hz: float = 100.0
    idle_rate_hz: float = 30.0
    input_ready_timeout_s: float = 3.0
    teleop_mapping: str = "relative_offset"
    pre_teleop_align_enabled: bool = True
    pre_teleop_align_error_limit_rad: float = 0.03
    reset_on_start: bool = False
    reset_after_episode: bool = True
    reset_timeout_s: float = 10.0
    reset_wait_s: float = 0.8
    reset_test_sample_time: int = 5
    reset_error_limit_rad: float = 0.02
    joint_step_limit_rad: float | None = 0.08
    idle_follow_enabled: bool = True
    control_mode: str = "position"
    mit: MitControlConfig = field(default_factory=MitControlConfig)
    role_switch_settle_s: float = 0.3
    role_switch_timeout_s: float = 3.0
    reset_interpolation_enabled: bool = True
    reset_interpolation_rate_hz: float = 30.0
    reset_joint_speed_rad_s: float = 1.0
    reset_min_duration_s: float = 0.2
    reset_max_step_rad: float = 0.05


@dataclass(frozen=True)
class TeleopConfig:
    mode: str = "master_slave"
    protocol: str = "can"
    backend: str = "pyagxarm"
    master_slave: tuple[ArmPairConfig, ...] = ()
    command: CommandConfig = field(default_factory=CommandConfig)


@dataclass(frozen=True)
class GripperConfig:
    enabled: bool = True
    effector: str = "AGX_GRIPPER"
    attach_to: str = "follower"
    teleop_enabled: bool = False
    scale: float = 1.0
    offset_m: float = 0.0
    offset_deg: float = 0.0
    min_width_m: float = 0.0
    max_width_m: float = 0.07
    force_n: float = 1.0
    command_rate_hz: float = 30.0
    deadband_m: float = 0.0005
    deadband_deg: float = 0.1
    keepalive_s: float = 0.5


@dataclass(frozen=True)
class CollectionConfig:
    teleop: TeleopConfig
    output: OutputConfig
    cameras: tuple[CameraConfig, ...] = ()
    gripper: GripperConfig = field(default_factory=GripperConfig)
    robot_states: dict[str, StateParamConfig] = field(default_factory=dict)
    raw_yaml: str = ""


DEFAULT_STATE_PARAMS = {
    "q": StateParamConfig(enabled=True, lowpass=False),
    "velocity": StateParamConfig(enabled=True, lowpass=False),
    "acceleration": StateParamConfig(enabled=False, lowpass=False),
    "ee_pose": StateParamConfig(enabled=True, lowpass=False),
    "torque": StateParamConfig(enabled=True, lowpass=False),
    "current": StateParamConfig(enabled=False, lowpass=False),
}


def load_config(path: str | Path) -> CollectionConfig:
    config_path = Path(path).expanduser().resolve()
    raw_yaml = config_path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw_yaml) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")

    teleop = _parse_teleop(data.get("teleop", {}))
    if teleop.mode not in SUPPORTED_TELEOP_MODES:
        raise ValueError(f"Unsupported teleop.mode={teleop.mode!r}; choose one of {sorted(SUPPORTED_TELEOP_MODES)}")
    if teleop.mode != "master_slave":
        raise NotImplementedError(f"teleop.mode={teleop.mode!r} is reserved but not implemented yet")
    if teleop.protocol != "can":
        raise NotImplementedError("Only CAN protocol is implemented in this first pass")
    if not teleop.master_slave:
        raise ValueError("teleop.master_slave.arm_pairs must contain at least one leader/follower pair")

    output = _parse_output(data.get("output", {}), config_path.parent)
    cameras = tuple(_parse_camera(item) for item in data.get("cameras", []) if item.get("enabled", True))
    gripper = _parse_gripper(data.get("gripper", {}))
    robot_states = _parse_state_params(data.get("robot_states", {}))
    return CollectionConfig(
        teleop=teleop,
        output=output,
        cameras=cameras,
        gripper=gripper,
        robot_states=robot_states,
        raw_yaml=raw_yaml,
    )


def _parse_teleop(data: dict[str, Any]) -> TeleopConfig:
    if not isinstance(data, dict):
        raise ValueError("teleop must be a mapping")
    master_slave_data = data.get("master_slave", {})
    arm_pairs_data = master_slave_data.get("arm_pairs", []) if isinstance(master_slave_data, dict) else []
    pairs = tuple(_parse_arm_pair(item) for item in arm_pairs_data)
    return TeleopConfig(
        mode=str(data.get("mode", "master_slave")),
        protocol=str(data.get("protocol", "can")),
        backend=str(data.get("backend", "pyagxarm")),
        master_slave=pairs,
        command=_parse_command(data.get("command", {})),
    )


def _parse_command(data: dict[str, Any]) -> CommandConfig:
    if not isinstance(data, dict):
        raise ValueError("teleop.command must be a mapping")
    control_mode = str(data.get("control_mode", "position")).lower()
    if control_mode not in {"position", "mit"}:
        raise ValueError("teleop.command.control_mode must be one of: position, mit")
    role_switch_settle_s = float(data.get("role_switch_settle_s", 0.3))
    role_switch_timeout_s = float(data.get("role_switch_timeout_s", 3.0))
    reset_interpolation_rate_hz = float(data.get("reset_interpolation_rate_hz", 30.0))
    reset_joint_speed_rad_s = float(data.get("reset_joint_speed_rad_s", 1.0))
    reset_min_duration_s = float(data.get("reset_min_duration_s", 0.2))
    reset_max_step_rad = float(data.get("reset_max_step_rad", 0.05))
    if role_switch_settle_s < 0:
        raise ValueError("teleop.command.role_switch_settle_s must be non-negative")
    if role_switch_timeout_s <= 0:
        raise ValueError("teleop.command.role_switch_timeout_s must be positive")
    if reset_interpolation_rate_hz <= 0:
        raise ValueError("teleop.command.reset_interpolation_rate_hz must be positive")
    if reset_joint_speed_rad_s <= 0:
        raise ValueError("teleop.command.reset_joint_speed_rad_s must be positive")
    if reset_min_duration_s < 0:
        raise ValueError("teleop.command.reset_min_duration_s must be non-negative")
    if reset_max_step_rad <= 0:
        raise ValueError("teleop.command.reset_max_step_rad must be positive")
    return CommandConfig(
        sample_rate_hz=float(data.get("sample_rate_hz", 100.0)),
        idle_rate_hz=float(data.get("idle_rate_hz", 30.0)),
        input_ready_timeout_s=float(data.get("input_ready_timeout_s", 3.0)),
        teleop_mapping=str(data.get("teleop_mapping", "relative_offset")),
        pre_teleop_align_enabled=bool(data.get("pre_teleop_align_enabled", True)),
        pre_teleop_align_error_limit_rad=float(data.get("pre_teleop_align_error_limit_rad", 0.03)),
        reset_on_start=bool(data.get("reset_on_start", False)),
        reset_after_episode=bool(data.get("reset_after_episode", True)),
        reset_timeout_s=float(data.get("reset_timeout_s", 10.0)),
        reset_wait_s=float(data.get("reset_wait_s", 0.8)),
        reset_test_sample_time=int(data.get("reset_test_sample_time", 5)),
        reset_error_limit_rad=float(data.get("reset_error_limit_rad", 0.02)),
        joint_step_limit_rad=_optional_float(data.get("joint_step_limit_rad", 0.08)),
        idle_follow_enabled=bool(data.get("idle_follow_enabled", True)),
        control_mode=control_mode,
        mit=_parse_mit_control(data.get("mit", {})),
        role_switch_settle_s=role_switch_settle_s,
        role_switch_timeout_s=role_switch_timeout_s,
        reset_interpolation_enabled=bool(data.get("reset_interpolation_enabled", True)),
        reset_interpolation_rate_hz=reset_interpolation_rate_hz,
        reset_joint_speed_rad_s=reset_joint_speed_rad_s,
        reset_min_duration_s=reset_min_duration_s,
        reset_max_step_rad=reset_max_step_rad,
    )


def _parse_mit_control(data: dict[str, Any]) -> MitControlConfig:
    if not isinstance(data, dict):
        raise ValueError("teleop.command.mit must be a mapping")
    kp = _joint_vector(data.get("kp", (5.0,) * 7), "kp")
    kd = _joint_vector(data.get("kd", (0.8,) * 7), "kd")
    v_des = _joint_vector(data.get("v_des", (0.0,) * 7), "v_des")
    t_ff = _joint_vector(data.get("t_ff", (0.0,) * 7), "t_ff")
    if any(value < 0.0 or value > 500.0 for value in kp):
        raise ValueError("teleop.command.mit.kp values must be within [0, 500]")
    if any(value < -5.0 or value > 5.0 for value in kd):
        raise ValueError("teleop.command.mit.kd values must be within [-5, 5]")
    if any(value < -45.0 or value > 45.0 for value in v_des):
        raise ValueError("teleop.command.mit.v_des values must be within [-45, 45] rad/s")
    torque_limits = (24.0, 24.0, 16.0, 16.0, 8.0, 8.0, 8.0)
    if any(abs(value) > limit for value, limit in zip(t_ff, torque_limits)):
        raise ValueError(
            "teleop.command.mit.t_ff exceeds Nero limits: "
            "J1-J2 +/-24, J3-J4 +/-16, J5-J7 +/-8 N.m"
        )
    return MitControlConfig(kp=kp, kd=kd, v_des=v_des, t_ff=t_ff)


def _parse_output(data: dict[str, Any], base_dir: Path) -> OutputConfig:
    if not isinstance(data, dict):
        raise ValueError("output must be a mapping")
    directory = Path(data.get("directory", "runs/nero_master_slave")).expanduser()
    if not directory.is_absolute():
        directory = (base_dir / directory).resolve()
    return OutputConfig(directory=directory, prefix=str(data.get("prefix", "episode")))


def _parse_camera(data: dict[str, Any]) -> CameraConfig:
    if not isinstance(data, dict):
        raise ValueError("Each camera entry must be a mapping")
    name = data.get("name")
    if not name:
        raise ValueError("Each enabled camera must define a name")
    known = {
        "name",
        "enabled",
        "backend",
        "serial_number",
        "width",
        "height",
        "fps",
        "exposure",
        "depth",
        "crop",
        "output_size",
    }
    crop = tuple(data.get("crop", (0, None, 0, None)))
    output_size = data.get("output_size")
    return CameraConfig(
        name=str(name),
        enabled=bool(data.get("enabled", True)),
        backend=str(data.get("backend", "orbbec_dabai")),
        serial_number=data.get("serial_number"),
        width=int(data.get("width", 640)),
        height=int(data.get("height", 480)),
        fps=float(data.get("fps", 30.0)),
        exposure=data.get("exposure"),
        depth=bool(data.get("depth", False)),
        crop=_normalize_crop(crop),
        output_size=tuple(output_size) if output_size else None,
        extra={key: value for key, value in data.items() if key not in known},
    )


def _parse_gripper(data: dict[str, Any]) -> GripperConfig:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("gripper must be a mapping")
    attach_to = str(data.get("attach_to", "follower"))
    if attach_to not in {"leader", "follower", "both"}:
        raise ValueError("gripper.attach_to must be one of: leader, follower, both")
    scale = float(data.get("scale", 1.0))
    offset_m = float(data.get("offset_m", 0.0))
    offset_deg = float(data.get("offset_deg", 0.0))
    min_width_m = float(data.get("min_width_m", 0.0))
    max_width_m = float(data.get("max_width_m", 0.07))
    force_n = float(data.get("force_n", 1.0))
    command_rate_hz = float(data.get("command_rate_hz", 30.0))
    deadband_m = float(data.get("deadband_m", 0.0005))
    deadband_deg = float(data.get("deadband_deg", 0.1))
    keepalive_s = float(data.get("keepalive_s", 0.5))
    numeric_values = (
        scale,
        offset_m,
        offset_deg,
        min_width_m,
        max_width_m,
        force_n,
        command_rate_hz,
        deadband_m,
        deadband_deg,
        keepalive_s,
    )
    if not all(isfinite(value) for value in numeric_values):
        raise ValueError("gripper numeric parameters must be finite")
    if scale == 0:
        raise ValueError("gripper.scale must be non-zero")
    if min_width_m < 0 or max_width_m <= min_width_m:
        raise ValueError("gripper width range must satisfy 0 <= min_width_m < max_width_m")
    if force_n < 0:
        raise ValueError("gripper.force_n must be non-negative")
    if command_rate_hz <= 0:
        raise ValueError("gripper.command_rate_hz must be positive")
    if deadband_m < 0:
        raise ValueError("gripper.deadband_m must be non-negative")
    if deadband_deg < 0:
        raise ValueError("gripper.deadband_deg must be non-negative")
    if keepalive_s <= 0:
        raise ValueError("gripper.keepalive_s must be positive")
    return GripperConfig(
        enabled=bool(data.get("enabled", True)),
        effector=str(data.get("effector", "AGX_GRIPPER")),
        attach_to=attach_to,
        teleop_enabled=bool(data.get("teleop_enabled", False)),
        scale=scale,
        offset_m=offset_m,
        offset_deg=offset_deg,
        min_width_m=min_width_m,
        max_width_m=max_width_m,
        force_n=force_n,
        command_rate_hz=command_rate_hz,
        deadband_m=deadband_m,
        deadband_deg=deadband_deg,
        keepalive_s=keepalive_s,
    )


def _parse_arm_pair(data: dict[str, Any]) -> ArmPairConfig:
    if not isinstance(data, dict):
        raise ValueError("Each arm pair must be a mapping")
    name = str(data.get("name", f"pair_{id(data):x}"))
    return ArmPairConfig(
        name=name,
        leader=_parse_arm_endpoint(data.get("leader", {}), f"{name}_leader"),
        follower=_parse_arm_endpoint(data.get("follower", {}), f"{name}_follower"),
    )


def _parse_arm_endpoint(data: dict[str, Any], default_name: str) -> ArmEndpointConfig:
    if not isinstance(data, dict):
        raise ValueError("leader/follower arm config must be a mapping")
    known = {
        "name",
        "can_id",
        "id",
        "channel",
        "interface",
        "bitrate",
        "firmware",
        "rest_q",
        "config_kwargs",
    }
    config_kwargs = dict(data.get("config_kwargs", {}))
    for key, value in data.items():
        if key not in known:
            config_kwargs.setdefault(key, value)
    can_id = data.get("can_id", data.get("id"))
    return ArmEndpointConfig(
        name=str(data.get("name", default_name)),
        can_id=int(can_id) if can_id is not None else None,
        channel=str(data.get("channel", "can0")),
        interface=str(data.get("interface", "socketcan")),
        bitrate=int(data.get("bitrate", 1_000_000)),
        firmware=str(data.get("firmware", "V1_6")),
        rest_q=tuple(float(x) for x in data.get("rest_q", ())),
        config_kwargs=config_kwargs,
    )


def _parse_state_params(data: dict[str, Any]) -> dict[str, StateParamConfig]:
    if not isinstance(data, dict):
        raise ValueError("robot_states must be a mapping")
    params = dict(DEFAULT_STATE_PARAMS)
    for name, value in data.items():
        params[str(name)] = _parse_state_param(value)
    return params


def _parse_state_param(value: Any) -> StateParamConfig:
    if isinstance(value, bool):
        return StateParamConfig(enabled=value, lowpass=False)
    if value is None:
        return StateParamConfig(enabled=True, lowpass=False)
    if not isinstance(value, dict):
        raise ValueError("Each robot_states item must be bool or mapping")
    velocity_lowpass_cutoff_hz = _optional_float(value.get("velocity_lowpass_cutoff_hz"))
    if velocity_lowpass_cutoff_hz is not None and velocity_lowpass_cutoff_hz <= 0:
        raise ValueError("velocity_lowpass_cutoff_hz must be positive when provided")
    return StateParamConfig(
        enabled=bool(value.get("enabled", True)),
        lowpass=bool(value.get("lowpass", False)),
        lowpass_cutoff_hz=float(value.get("lowpass_cutoff_hz", 12.0)),
        velocity_lowpass_cutoff_hz=velocity_lowpass_cutoff_hz,
    )


def _normalize_crop(crop: tuple[Any, ...]) -> tuple[int | None, int | None, int | None, int | None]:
    if len(crop) != 4:
        raise ValueError("camera.crop must contain four values: [y0, y1, x0, x1]")
    return tuple(None if value is None else int(value) for value in crop)  # type: ignore[return-value]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _joint_vector(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 7:
        raise ValueError(f"teleop.command.mit.{name} must contain exactly 7 values")
    vector = tuple(float(item) for item in value)
    if not all(isfinite(item) for item in vector):
        raise ValueError(f"teleop.command.mit.{name} values must be finite")
    return vector
