from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from calibration.common import (
    CalibrationPlan,
    CalibrationPose,
    StaticDataset,
    load_plan,
    metadata_for_plan,
    save_dataset,
    setup_socketcan,
)
from calibration.static_model import TerminalStaticModel
from nero_collection.arms.base import ArmState
from nero_collection.arms.factory import build_arm
from nero_collection.config import ArmEndpointConfig, load_config


log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    plan = load_plan(args.config)
    model = TerminalStaticModel(plan.model)
    validate_pose_plan(plan, model)
    _print_pose_plan(plan)
    if args.plan_only:
        print("Plan-only check passed. No CAN connection or motion was attempted.")
        return 0
    if not plan.safety.approved:
        raise RuntimeError(
            "calibration safety.approved is false; inspect every pose on the real workcell first"
        )

    collection = load_config(plan.collection_config_path)
    pair_name = args.pair or plan.pair_name
    endpoint = _follower_endpoint(collection, pair_name)
    backend = args.backend or collection.teleop.backend
    if backend == "pyagxarm" and not args.skip_can_setup:
        setup_socketcan(endpoint.channel, endpoint.bitrate)
    if not args.yes:
        _confirm_motion(endpoint, plan)

    output = Path(args.output).expanduser().resolve() if args.output else _default_output(plan, args.label)
    arm = build_arm(endpoint, backend)
    q_samples: list[np.ndarray] = []
    tau_samples: list[np.ndarray] = []
    current_samples: list[np.ndarray] = []
    timestamps: list[int] = []
    pose_indices: list[int] = []
    round_indices: list[int] = []

    log.info("connecting follower arm=%s channel=%s backend=%s", endpoint.name, endpoint.channel, backend)
    arm.connect()
    try:
        arm.set_follower_mode()
        time.sleep(collection.teleop.command.role_switch_settle_s)
        arm.enable()
        _wait_for_follower_role(
            arm,
            timeout_s=collection.teleop.command.role_switch_timeout_s,
        )
        for round_index in range(plan.motion.round_count):
            reverse = plan.motion.alternate_reverse and round_index % 2 == 1
            pose_order = list(range(len(plan.poses)))
            if reverse:
                pose_order.reverse()
            log.info(
                "starting calibration round %d/%d order=%s",
                round_index + 1,
                plan.motion.round_count,
                "reverse" if reverse else "forward",
            )
            for order_index, pose_index in enumerate(pose_order):
                pose = plan.poses[pose_index]
                log.info(
                    "moving round=%d pose=%d/%d name=%s",
                    round_index + 1,
                    order_index + 1,
                    len(plan.poses),
                    pose.name,
                )
                _move_to_pose(arm, pose, plan)
                _wait_until_static(arm, pose, plan)
                samples = _sample_static_pose(arm, pose, plan)
                for state in samples:
                    q_samples.append(state.q.copy())
                    tau_samples.append(state.torque.copy())
                    current_samples.append(state.current.copy())
                    timestamps.append(int(state.timestamp_us))
                    pose_indices.append(pose_index)
                    round_indices.append(round_index)
                median_tau = np.median(np.stack([state.torque for state in samples]), axis=0)
                log.info(
                    "sampled round=%d pose=%s samples=%d median_tau_nm=%s",
                    round_index + 1,
                    pose.name,
                    len(samples),
                    np.array2string(median_tau, precision=4, suppress_small=True),
                )
    finally:
        # Do not disable a gravity-loaded arm automatically on exit.
        arm.disconnect()

    metadata = metadata_for_plan(plan)
    metadata.update(
        {
            "label": args.label,
            "arm_name": endpoint.name,
            "channel": endpoint.channel,
            "backend": backend,
            "completed": True,
            "round_count": plan.motion.round_count,
            "alternate_reverse": plan.motion.alternate_reverse,
            "created_at": datetime.now().astimezone().isoformat(),
        }
    )
    dataset = StaticDataset(
        q=np.stack(q_samples),
        tau=np.stack(tau_samples),
        current=np.stack(current_samples),
        timestamp_us=np.asarray(timestamps, dtype=np.int64),
        pose_index=np.asarray(pose_indices, dtype=np.int32),
        round_index=np.asarray(round_indices, dtype=np.int32),
        pose_names=tuple(pose.name for pose in plan.poses),
        metadata=metadata,
    )
    saved = save_dataset(output, dataset)
    print(f"Saved static calibration dataset: {saved}")
    return 0


def validate_pose_plan(plan: CalibrationPlan, model: TerminalStaticModel) -> None:
    lower = model.lower_position_limit + plan.safety.joint_limit_margin_rad
    upper = model.upper_position_limit - plan.safety.joint_limit_margin_rad
    if np.any(lower >= upper):
        raise ValueError("joint_limit_margin_rad leaves an empty joint range")
    for pose in plan.poses:
        if np.any(pose.q < lower) or np.any(pose.q > upper):
            invalid = np.flatnonzero((pose.q < lower) | (pose.q > upper)) + 1
            raise ValueError(
                f"pose {pose.name!r} violates URDF joint limits plus safety margin on joints {invalid.tolist()}"
            )


def _move_to_pose(arm, pose: CalibrationPose, plan: CalibrationPlan) -> None:
    start = _checked_state(arm.read_state(), plan, context=f"before pose {pose.name}")
    delta = pose.q - start.q
    max_delta = float(np.max(np.abs(delta)))
    # Quintic minimum-jerk scaling has max ds/du = 1.875. Account for that
    # factor so joint_speed_rad_s is a peak speed limit, not an average speed.
    duration_s = 1.875 * max_delta / plan.motion.joint_speed_rad_s
    steps = max(
        1,
        int(np.ceil(duration_s * plan.motion.command_rate_hz)),
        int(np.ceil(1.875 * max_delta / plan.motion.max_step_rad)),
    )
    start_t = time.monotonic()
    for step in range(1, steps + 1):
        phase = step / steps
        alpha = _minimum_jerk_scale(phase)
        arm.move_joints(start.q + alpha * delta)
        state = _checked_state(arm.read_state(), plan, context=f"moving to {pose.name}")
        if np.max(np.abs(state.dq)) > plan.motion.max_motion_velocity_rad_s:
            raise RuntimeError(f"velocity safety limit exceeded while moving to {pose.name}: {state.dq}")
        next_t = start_t + step / plan.motion.command_rate_hz
        remaining = next_t - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
    if not arm.wait_motion_done(plan.motion.motion_timeout_s):
        raise RuntimeError(f"motion timeout at calibration pose {pose.name}")


def _minimum_jerk_scale(phase: float) -> float:
    phase = float(np.clip(phase, 0.0, 1.0))
    return phase**3 * (10.0 - 15.0 * phase + 6.0 * phase**2)


def _wait_until_static(arm, pose: CalibrationPose, plan: CalibrationPlan) -> None:
    deadline = time.monotonic() + plan.motion.settle_timeout_s
    stable_since: float | None = None
    period = 1.0 / plan.motion.sample_rate_hz
    while time.monotonic() < deadline:
        state = _checked_state(arm.read_state(), plan, context=f"settling at {pose.name}")
        velocity_ok = float(np.max(np.abs(state.dq))) <= plan.motion.max_static_velocity_rad_s
        position_ok = float(np.max(np.abs(state.q - pose.q))) <= plan.motion.max_position_error_rad
        now = time.monotonic()
        if velocity_ok and position_ok:
            stable_since = now if stable_since is None else stable_since
            if now - stable_since >= plan.motion.settle_duration_s:
                return
        else:
            stable_since = None
        time.sleep(period)
    raise RuntimeError(f"arm did not settle at pose {pose.name} before timeout")


def _sample_static_pose(arm, pose: CalibrationPose, plan: CalibrationPlan) -> list[ArmState]:
    count = max(1, int(np.ceil(plan.motion.sample_duration_s * plan.motion.sample_rate_hz)))
    period = 1.0 / plan.motion.sample_rate_hz
    start_t = time.monotonic()
    samples: list[ArmState] = []
    for sample_index in range(count):
        state = _checked_state(arm.read_state(), plan, context=f"sampling {pose.name}")
        if float(np.max(np.abs(state.dq))) > plan.motion.max_static_velocity_rad_s:
            raise RuntimeError(f"arm moved during static sampling at pose {pose.name}: dq={state.dq}")
        if float(np.max(np.abs(state.q - pose.q))) > plan.motion.max_position_error_rad:
            raise RuntimeError(f"arm left target during static sampling at pose {pose.name}: q={state.q}")
        samples.append(state)
        next_t = start_t + (sample_index + 1) * period
        remaining = next_t - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
    return samples


def _checked_state(state: ArmState, plan: CalibrationPlan, *, context: str) -> ArmState:
    for name, values in (("q", state.q), ("dq", state.dq), ("tau", state.torque)):
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if vector.size != 7 or not np.isfinite(vector).all():
            raise RuntimeError(f"invalid {name} {context}: {vector}")
    torque = np.asarray(state.torque, dtype=np.float64)
    if np.any(np.abs(torque) > plan.safety.max_abs_torque_nm):
        raise RuntimeError(
            f"torque safety limit exceeded {context}: tau={torque}, limit={plan.safety.max_abs_torque_nm}"
        )
    current = np.asarray(state.current, dtype=np.float64).reshape(-1)
    if current.size != 7:
        current = np.full(7, np.nan, dtype=np.float64)
    return ArmState(
        q=np.asarray(state.q, dtype=np.float64).copy(),
        dq=np.asarray(state.dq, dtype=np.float64).copy(),
        ddq=np.asarray(state.ddq, dtype=np.float64).copy(),
        ee_pose=np.asarray(state.ee_pose, dtype=np.float64).copy(),
        torque=torque.copy(),
        current=current.copy(),
        timestamp_us=int(state.timestamp_us),
    )


def _wait_for_follower_role(arm, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_role = None
    while time.monotonic() < deadline:
        last_role = arm.read_control_role(refresh=True)
        if last_role == "follower":
            return
        time.sleep(0.05)
    raise RuntimeError(f"follower mode was not confirmed; detected={last_role or 'unknown'}")


def _follower_endpoint(collection, pair_name: str) -> ArmEndpointConfig:
    for pair in collection.teleop.master_slave:
        if pair.name == pair_name:
            return pair.follower
    raise ValueError(f"pair {pair_name!r} not found in collection config")


def _confirm_motion(endpoint: ArmEndpointConfig, plan: CalibrationPlan) -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("interactive confirmation requires a TTY; use --yes only after reviewing the plan")
    print(
        f"WARNING: {endpoint.name} on {endpoint.channel} will move through "
        f"{len(plan.poses)} poses for {plan.motion.round_count} rounds.\n"
        "Clear the workspace, keep the emergency stop ready, and keep the gripper empty."
    )
    answer = input("Type MOVE to continue: ").strip()
    if answer != "MOVE":
        raise RuntimeError("calibration cancelled")


def _print_pose_plan(plan: CalibrationPlan) -> None:
    print("Static gravity calibration pose plan:")
    for index, pose in enumerate(plan.poses):
        values = ", ".join(f"{value:+.5f}" for value in pose.q)
        print(f"  {index:02d} {pose.name:<16} [{values}]")
    print(f"safety.approved={plan.safety.approved}")
    print(
        f"rounds={plan.motion.round_count} "
        f"alternate_reverse={plan.motion.alternate_reverse}"
    )


def _default_output(plan: CalibrationPlan, label: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return plan.source_path.parent / "data" / f"static_{label}_{stamp}.npz"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect contact-free static gravity data from the Nero follower arm.")
    parser.add_argument("--config", default="calibration/config.yaml")
    parser.add_argument("--pair", help="Override the pair selected in calibration config")
    parser.add_argument("--backend", choices=("pyagxarm", "mock"))
    parser.add_argument("--output")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--plan-only", action="store_true", help="Validate and print poses without connecting CAN")
    parser.add_argument("--skip-can-setup", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip the final interactive MOVE confirmation")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
