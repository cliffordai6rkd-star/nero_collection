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
    median_window: int = 1
    velocity_lowpass_cutoff_hz: float | None = None


@dataclass(frozen=True)
class OutputConfig:
    directory: Path
    prefix: str = "episode"


@dataclass(frozen=True)
class DynamicsProcessingConfig:
    enabled: bool = False
    state_method: str = "spline"
    spline_smoothing_rad2: float = 1.0e-5
    fourier_fundamental_hz: float = 0.1
    fourier_harmonics: int = 8
    torque_lowpass_hz: float = 12.0
    torque_median_window: int = 3
    min_samples: int = 20


@dataclass(frozen=True)
class ContactWrenchConfig:
    urdf_path: Path = Path("urdf/nero/nero_with_gripper.urdf")
    frame_name: str = "gripper_base"
    delay_s: float = 0.5
    damping: float = 0.02
    reference_frame: str = "local"
    locked_joint_names: tuple[str, ...] = (
        "gripper",
        "gripper_joint1",
        "gripper_joint2",
    )
    gravity_m_s2: tuple[float, float, float] = (0.0, 0.0, -9.81)


@dataclass(frozen=True)
class InverseDynamicsConfig:
    urdf_path: Path = Path("urdf/nero/nero_with_gripper.urdf")
    manifest_path: Path | None = None
    delay_s: float = 0.5
    locked_joint_names: tuple[str, ...] = (
        "gripper",
        "gripper_joint1",
        "gripper_joint2",
    )
    gravity_m_s2: tuple[float, float, float] = (0.0, 0.0, -9.81)


@dataclass(frozen=True)
class RealtimePlotConfig:
    enabled: bool = False
    window_s: float = 10.0
    update_rate_hz: float = 20.0
    inverse_dynamics: InverseDynamicsConfig = field(default_factory=InverseDynamicsConfig)


@dataclass(frozen=True)
class CameraConfig:
    name: str
    enabled: bool = True
    backend: str = "orbbec_dabai"
    device: str | int | None = None
    pixel_format: str = "MJPG"
    buffer_size: int = 1
    startup_timeout_s: float = 3.0
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
    min_width_m: float = 0.0
    max_width_m: float = 0.07
    force_n: float = 1.0
    command_rate_hz: float = 30.0
    deadband_m: float = 0.0005
    keepalive_s: float = 0.5


@dataclass(frozen=True)
class CollectionConfig:
    teleop: TeleopConfig
    output: OutputConfig
    cameras: tuple[CameraConfig, ...] = ()
    gripper: GripperConfig = field(default_factory=GripperConfig)
    realtime_plot: RealtimePlotConfig = field(default_factory=RealtimePlotConfig)
    dynamics_processing: DynamicsProcessingConfig = field(default_factory=DynamicsProcessingConfig)
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
    realtime_plot = _parse_realtime_plot(data.get("realtime_plot", {}), config_path.parent)
    dynamics_processing = _parse_dynamics_processing(data.get("dynamics_processing", {}))
    robot_states = _parse_state_params(data.get("robot_states", {}))
    return CollectionConfig(
        teleop=teleop,
        output=output,
        cameras=cameras,
        gripper=gripper,
        realtime_plot=realtime_plot,
        dynamics_processing=dynamics_processing,
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


def _parse_dynamics_processing(data: dict[str, Any]) -> DynamicsProcessingConfig:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("dynamics_processing must be a mapping")
    state_method = str(data.get("state_method", "spline")).lower()
    if state_method not in {"spline", "fourier"}:
        raise ValueError("dynamics_processing.state_method must be spline or fourier")
    smoothing = float(data.get("spline_smoothing_rad2", 1.0e-5))
    fundamental_hz = float(data.get("fourier_fundamental_hz", 0.1))
    harmonics = int(data.get("fourier_harmonics", 8))
    torque_lowpass_hz = float(data.get("torque_lowpass_hz", 12.0))
    torque_median_window = int(data.get("torque_median_window", 3))
    min_samples = int(data.get("min_samples", 20))
    if not isfinite(smoothing) or smoothing < 0:
        raise ValueError("dynamics_processing.spline_smoothing_rad2 must be non-negative and finite")
    if not isfinite(fundamental_hz) or fundamental_hz <= 0:
        raise ValueError("dynamics_processing.fourier_fundamental_hz must be positive and finite")
    if harmonics < 1:
        raise ValueError("dynamics_processing.fourier_harmonics must be positive")
    if not isfinite(torque_lowpass_hz) or torque_lowpass_hz <= 0:
        raise ValueError("dynamics_processing.torque_lowpass_hz must be positive and finite")
    if torque_median_window < 1 or torque_median_window % 2 == 0:
        raise ValueError("dynamics_processing.torque_median_window must be a positive odd integer")
    if min_samples < 4:
        raise ValueError("dynamics_processing.min_samples must be at least 4")
    return DynamicsProcessingConfig(
        enabled=bool(data.get("enabled", False)),
        state_method=state_method,
        spline_smoothing_rad2=smoothing,
        fourier_fundamental_hz=fundamental_hz,
        fourier_harmonics=harmonics,
        torque_lowpass_hz=torque_lowpass_hz,
        torque_median_window=torque_median_window,
        min_samples=min_samples,
    )


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
        "device",
        "pixel_format",
        "buffer_size",
        "startup_timeout_s",
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
    device = data.get("device")
    if device is not None and not isinstance(device, (str, int)):
        raise ValueError("camera.device must be a device path or integer index")
    backend = str(data.get("backend", "orbbec_dabai"))
    normalized_backend = backend.lower().replace("-", "_")
    serial_number_value = data.get("serial_number")
    serial_number = (
        str(serial_number_value).strip() if serial_number_value is not None else None
    )
    if serial_number == "":
        raise ValueError("camera.serial_number must be non-empty")
    if normalized_backend in {"v4l2", "opencv_v4l2", "opencv"}:
        if device is None and serial_number is None:
            raise ValueError("V4L2 camera configuration must define device or serial_number")
        if device is not None and serial_number is not None:
            raise ValueError("V4L2 camera configuration must not define both device and serial_number")
    pixel_format = str(data.get("pixel_format", "MJPG")).upper()
    if len(pixel_format) != 4 or not pixel_format.isascii():
        raise ValueError("camera.pixel_format must be a four-character V4L2 code")
    width = int(data.get("width", 640))
    height = int(data.get("height", 480))
    fps = float(data.get("fps", 30.0))
    buffer_size = int(data.get("buffer_size", 1))
    startup_timeout_s = float(data.get("startup_timeout_s", 3.0))
    if width <= 0 or height <= 0:
        raise ValueError("camera width and height must be positive")
    if not isfinite(fps) or fps <= 0:
        raise ValueError("camera fps must be positive and finite")
    if buffer_size <= 0:
        raise ValueError("camera.buffer_size must be positive")
    if not isfinite(startup_timeout_s) or startup_timeout_s <= 0:
        raise ValueError("camera.startup_timeout_s must be positive and finite")
    normalized_crop = _normalize_crop(crop)
    normalized_output_size = tuple(int(value) for value in output_size) if output_size else None
    if normalized_output_size is not None and (
        len(normalized_output_size) != 2 or any(value <= 0 for value in normalized_output_size)
    ):
        raise ValueError("camera.output_size must contain positive [width, height]")
    return CameraConfig(
        name=str(name),
        enabled=bool(data.get("enabled", True)),
        backend=backend,
        device=device,
        pixel_format=pixel_format,
        buffer_size=buffer_size,
        startup_timeout_s=startup_timeout_s,
        serial_number=serial_number,
        width=width,
        height=height,
        fps=fps,
        exposure=data.get("exposure"),
        depth=bool(data.get("depth", False)),
        crop=normalized_crop,
        output_size=normalized_output_size,
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
    min_width_m = float(data.get("min_width_m", 0.0))
    max_width_m = float(data.get("max_width_m", 0.07))
    force_n = float(data.get("force_n", 1.0))
    command_rate_hz = float(data.get("command_rate_hz", 30.0))
    deadband_m = float(data.get("deadband_m", 0.0005))
    keepalive_s = float(data.get("keepalive_s", 0.5))
    numeric_values = (
        scale,
        offset_m,
        min_width_m,
        max_width_m,
        force_n,
        command_rate_hz,
        deadband_m,
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
    if keepalive_s <= 0:
        raise ValueError("gripper.keepalive_s must be positive")
    return GripperConfig(
        enabled=bool(data.get("enabled", True)),
        effector=str(data.get("effector", "AGX_GRIPPER")),
        attach_to=attach_to,
        teleop_enabled=bool(data.get("teleop_enabled", False)),
        scale=scale,
        offset_m=offset_m,
        min_width_m=min_width_m,
        max_width_m=max_width_m,
        force_n=force_n,
        command_rate_hz=command_rate_hz,
        deadband_m=deadband_m,
        keepalive_s=keepalive_s,
    )


def _parse_realtime_plot(
    data: dict[str, Any],
    config_dir: Path | None = None,
) -> RealtimePlotConfig:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("realtime_plot must be a mapping")
    window_s = float(data.get("window_s", 10.0))
    update_rate_hz = float(data.get("update_rate_hz", 20.0))
    if not isfinite(window_s) or window_s <= 0:
        raise ValueError("realtime_plot.window_s must be positive and finite")
    if not isfinite(update_rate_hz) or update_rate_hz <= 0:
        raise ValueError("realtime_plot.update_rate_hz must be positive and finite")
    if "inverse_dynamics" in data and "contact_wrench" in data:
        raise ValueError("realtime_plot must not define both inverse_dynamics and contact_wrench")
    inverse_data = data.get("inverse_dynamics", data.get("contact_wrench", {}))
    if inverse_data is None:
        inverse_data = {}
    if not isinstance(inverse_data, dict):
        raise ValueError("realtime_plot.inverse_dynamics must be a mapping")
    base_dir = Path.cwd() if config_dir is None else Path(config_dir)
    urdf_path = Path(
        inverse_data.get("urdf_path", "../urdf/nero/nero_with_gripper.urdf")
    ).expanduser()
    if not urdf_path.is_absolute():
        urdf_path = (base_dir / urdf_path).resolve()
    manifest_value = inverse_data.get("manifest_path")
    manifest_path = None
    if manifest_value is not None:
        manifest_path = Path(manifest_value).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = (base_dir / manifest_path).resolve()
    delay_s = float(inverse_data.get("delay_s", 0.5))
    if not isfinite(delay_s) or delay_s < 0:
        raise ValueError("realtime_plot.inverse_dynamics.delay_s must be non-negative and finite")
    locked_joint_names = tuple(
        str(name) for name in inverse_data.get(
            "locked_joint_names",
            ("gripper", "gripper_joint1", "gripper_joint2"),
        )
    )
    if not all(locked_joint_names):
        raise ValueError("realtime_plot.inverse_dynamics.locked_joint_names must contain valid names")
    gravity = tuple(float(value) for value in inverse_data.get("gravity_m_s2", (0.0, 0.0, -9.81)))
    if len(gravity) != 3 or not all(isfinite(value) for value in gravity):
        raise ValueError("realtime_plot.inverse_dynamics.gravity_m_s2 must contain three finite values")
    return RealtimePlotConfig(
        enabled=bool(data.get("enabled", False)),
        window_s=window_s,
        update_rate_hz=update_rate_hz,
        inverse_dynamics=InverseDynamicsConfig(
            urdf_path=urdf_path,
            manifest_path=manifest_path,
            delay_s=delay_s,
            locked_joint_names=locked_joint_names,
            gravity_m_s2=gravity,
        ),
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
    lowpass_cutoff_hz = float(value.get("lowpass_cutoff_hz", 12.0))
    if not isfinite(lowpass_cutoff_hz) or lowpass_cutoff_hz <= 0:
        raise ValueError("lowpass_cutoff_hz must be positive and finite")
    median_window = int(value.get("median_window", 1))
    if median_window < 1 or median_window % 2 == 0:
        raise ValueError("median_window must be a positive odd integer")
    velocity_lowpass_cutoff_hz = _optional_float(value.get("velocity_lowpass_cutoff_hz"))
    if velocity_lowpass_cutoff_hz is not None and (
        not isfinite(velocity_lowpass_cutoff_hz) or velocity_lowpass_cutoff_hz <= 0
    ):
        raise ValueError("velocity_lowpass_cutoff_hz must be positive and finite when provided")
    return StateParamConfig(
        enabled=bool(value.get("enabled", True)),
        lowpass=bool(value.get("lowpass", False)),
        lowpass_cutoff_hz=lowpass_cutoff_hz,
        median_window=median_window,
        velocity_lowpass_cutoff_hz=velocity_lowpass_cutoff_hz,
    )


def _normalize_crop(crop: tuple[Any, ...]) -> tuple[int | None, int | None, int | None, int | None]:
    if len(crop) != 4:
        raise ValueError("camera.crop must contain four values: [y0, y1, x0, x1]")
    normalized = tuple(None if value is None else int(value) for value in crop)
    if any(value is not None and value < 0 for value in normalized):
        raise ValueError("camera.crop values must be non-negative or null")
    y0, y1, x0, x1 = normalized
    if y0 is not None and y1 is not None and y1 <= y0:
        raise ValueError("camera.crop y1 must be greater than y0")
    if x0 is not None and x1 is not None and x1 <= x0:
        raise ValueError("camera.crop x1 must be greater than x0")
    return normalized  # type: ignore[return-value]


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
