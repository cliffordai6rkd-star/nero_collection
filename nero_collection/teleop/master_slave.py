from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import numpy as np

from nero_collection.arms.base import ArmInterface, ArmState
from nero_collection.arms.factory import build_arm
from nero_collection.config import CollectionConfig
from nero_collection.time_utils import now_us

log = logging.getLogger(__name__)


@dataclass
class ArmPairRuntime:
    name: str
    leader: ArmInterface
    follower: ArmInterface
    rest_q_leader: np.ndarray
    rest_q_follower: np.ndarray


class MasterSlaveTeleop:
    def __init__(self, config: CollectionConfig) -> None:
        self.config = config
        backend = config.teleop.backend
        self.pairs = tuple(
            ArmPairRuntime(
                name=pair.name,
                leader=build_arm(pair.leader, backend),
                follower=build_arm(pair.follower, backend),
                rest_q_leader=_rest_q(pair.leader.rest_q),
                rest_q_follower=_rest_q(pair.follower.rest_q),
            )
            for pair in config.teleop.master_slave
        )
        self.arm_names = tuple(pair.name for pair in self.pairs)
        self._teleop_reference: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._hold_after_reset = False
        self._parked = False
        self._unrecorded_teleop = False
        self._last_gripper_command: dict[str, float] = {}
        self._last_gripper_command_mode: dict[str, str] = {}
        self._last_gripper_command_t: dict[str, float] = {}
        self._gripper_command_announced: set[str] = set()
        self._gripper_feedback_warned: set[str] = set()
        self._leader_gripper_feedback_timestamp_us: dict[str, int] = {}
        self._leader_gripper_feedback_change_t: dict[str, float] = {}
        self._leader_gripper_stale_warned: set[str] = set()

    def start(self) -> None:
        log.info("starting Nero master-slave arms over CAN")
        for pair in self.pairs:
            log.info("starting pair=%s leader=%s follower=%s", pair.name, pair.leader.name, pair.follower.name)
            pair.leader.connect()
            pair.follower.connect()
            if self.config.teleop.command.control_mode == "mit":
                pair.follower.validate_joint_impedance_support()
            self._log_current_roles(pair)
            self._prepare_pair_for_reset(pair)
            self._init_grippers(pair)
        self.check_input_devices()
        if self.config.teleop.command.reset_on_start:
            log.info("startup reset enabled: both arms reset to follower rest_q")
            self.reset_to_rest()
        else:
            self._set_parked_state()

    def shutdown(self) -> None:
        for pair in self.pairs:
            for arm in (pair.leader, pair.follower):
                try:
                    arm.disconnect()
                except Exception as exc:  # pragma: no cover - shutdown guard
                    log.debug("disconnect failed for %s: %s", arm.name, exc)

    def check_input_devices(self) -> None:
        log.info("checking master-slave teleop input devices")
        timeout_s = self.config.teleop.command.input_ready_timeout_s
        for pair in self.pairs:
            leader_q = self._wait_for_valid_joints(
                lambda: pair.leader.read_state().q,
                timeout_s,
                f"Leader endpoint {pair.leader.name}",
            )
            follower_q = self._wait_for_valid_joints(
                lambda: pair.follower.read_state().q,
                timeout_s,
                f"Follower arm {pair.follower.name}",
            )
            log.info("input ok pair=%s dof=%d", pair.name, leader_q.size)

    def _init_grippers(self, pair: ArmPairRuntime) -> None:
        gripper = self.config.gripper
        if not gripper.enabled:
            return
        if gripper.teleop_enabled or gripper.attach_to in {"leader", "both"}:
            pair.leader.init_gripper(gripper.effector)
            if gripper.teleop_enabled:
                pair.leader.disable_gripper()
        if gripper.teleop_enabled or gripper.attach_to in {"follower", "both"}:
            pair.follower.init_gripper(gripper.effector)

    def _wait_for_valid_joints(self, read_fn, timeout_s: float, label: str) -> np.ndarray:
        deadline = time.monotonic() + timeout_s
        last_q = np.empty((0,), dtype=np.float64)
        while time.monotonic() < deadline:
            last_q = np.asarray(read_fn(), dtype=np.float64).reshape(-1)
            if _is_valid_joint_vector(last_q):
                return last_q
            time.sleep(0.05)
        raise RuntimeError(f"{label} did not return valid joint positions within {timeout_s:.1f}s; last={last_q}")

    def enter_idle_follow(self) -> None:
        log.info("entering idle mode: leader arms are read-only, follower arms hold/follow by config")
        self._parked = False
        self._unrecorded_teleop = False
        self._teleop_reference.clear()
        for pair in self.pairs:
            if self.config.teleop.command.idle_follow_enabled and not self._hold_after_reset:
                self._ensure_teleop_reference(pair)

    def idle_step(self) -> None:
        if self._parked:
            return
        if not self._unrecorded_teleop and (
            not self.config.teleop.command.idle_follow_enabled or self._hold_after_reset
        ):
            return
        for pair in self.pairs:
            leader_target = pair.leader.read_leader_joint_positions()
            if not _is_valid_joint_vector(leader_target):
                continue
            follower_q = pair.follower.read_state().q
            follower_target = self._map_leader_to_follower(pair, leader_target)
            target = _limit_joint_step(follower_target, follower_q, self.config.teleop.command.joint_step_limit_rad)
            self._command_follower(pair, target)
        if self._unrecorded_teleop:
            self._update_gripper_teleop({})

    def enter_teleop(self) -> None:
        log.info("entering teleop mode: follower arms follow leader arms")
        for pair in self.pairs:
            self._prepare_pair_for_teleop(pair)
            self._ensure_teleop_reference(pair)
            log.info("teleop reference pair=%s mapping=%s", pair.name, self.config.teleop.command.teleop_mapping)
        self._hold_after_reset = False
        self._parked = False
        self._unrecorded_teleop = False

    def enter_unrecorded_teleop(self) -> None:
        if self._unrecorded_teleop:
            return
        log.info("entering teleoperation without recording")
        for pair in self.pairs:
            self._prepare_pair_for_teleop(pair)
            self._ensure_teleop_reference(pair)
        self._hold_after_reset = False
        self._parked = False
        self._unrecorded_teleop = True

    @staticmethod
    def _log_current_roles(pair: ArmPairRuntime) -> None:
        leader_role = pair.leader.read_control_role(refresh=True)
        follower_role = pair.follower.read_control_role(refresh=True)
        log.info(
            "current arm roles pair=%s leader_endpoint=%s:%s follower_endpoint=%s:%s",
            pair.name,
            pair.leader.name,
            leader_role or "unknown",
            pair.follower.name,
            follower_role or "unknown",
        )

    @staticmethod
    def _ensure_arm_role(pair_name: str, arm: ArmInterface, expected_role: str) -> bool:
        # Prefer roles commanded by this process. Leader mode can stop active CAN
        # status push, so a forced refresh may return the stale pre-switch mode.
        detected_role = arm.read_control_role()
        if detected_role == expected_role:
            log.info(
                "arm role matches required state pair=%s arm=%s role=%s",
                pair_name,
                arm.name,
                expected_role,
            )
            return False
        log.warning(
            "arm role mismatch pair=%s arm=%s detected=%s required=%s; switching",
            pair_name,
            arm.name,
            detected_role or "unknown",
            expected_role,
        )
        if expected_role == "leader":
            arm.set_leader_mode()
        elif expected_role == "follower":
            arm.set_follower_mode()
        else:  # pragma: no cover - internal contract
            raise ValueError(f"Unsupported arm role {expected_role!r}")
        log.info(
            "arm role switch command sent pair=%s arm=%s required_role=%s; awaiting hardware feedback",
            pair_name,
            arm.name,
            expected_role,
        )
        return True

    def _settle_after_role_switch(self, switched: bool) -> None:
        if switched and self.config.teleop.command.role_switch_settle_s > 0:
            time.sleep(self.config.teleop.command.role_switch_settle_s)

    def _wait_for_arm_role(self, pair_name: str, arm: ArmInterface, expected_role: str) -> None:
        timeout_s = self.config.teleop.command.role_switch_timeout_s
        deadline = time.monotonic() + timeout_s
        last_role: str | None = None
        while time.monotonic() < deadline:
            last_role = arm.read_control_role(refresh=True)
            if last_role == expected_role:
                log.info(
                    "arm role verified from hardware pair=%s arm=%s role=%s",
                    pair_name,
                    arm.name,
                    expected_role,
                )
                return
            time.sleep(0.05)
        raise RuntimeError(
            f"Arm role switch not confirmed for {arm.name}: "
            f"expected={expected_role}, detected={last_role or 'unknown'}"
        )

    def _prepare_pair_for_teleop(self, pair: ArmPairRuntime) -> None:
        switched = False
        switched |= self._ensure_arm_role(pair.name, pair.follower, "follower")
        switched |= self._ensure_arm_role(pair.name, pair.leader, "leader")
        self._settle_after_role_switch(switched)

        try:
            self._wait_for_arm_role(pair.name, pair.follower, "follower")
        except RuntimeError:
            log.warning(
                "retrying follower mode before teleop pair=%s arm=%s",
                pair.name,
                pair.follower.name,
            )
            pair.follower.set_follower_mode()
            self._settle_after_role_switch(True)
            self._wait_for_arm_role(pair.name, pair.follower, "follower")

        try:
            self._wait_for_arm_role(pair.name, pair.leader, "leader")
        except RuntimeError:
            log.warning(
                "retrying leader mode before teleop pair=%s arm=%s",
                pair.name,
                pair.leader.name,
            )
            pair.leader.set_leader_mode()
            self._settle_after_role_switch(True)
            self._wait_for_arm_role(pair.name, pair.leader, "leader")
        log.info(
            "leader mode verified from fresh leader feedback pair=%s arm=%s",
            pair.name,
            pair.leader.name,
        )

    def _prepare_pair_for_reset(self, pair: ArmPairRuntime) -> None:
        switched = False
        switched |= self._ensure_arm_role(pair.name, pair.leader, "follower")
        switched |= self._ensure_arm_role(pair.name, pair.follower, "follower")
        self._settle_after_role_switch(switched)
        log.info("confirming both arms enabled before reset pair=%s", pair.name)
        pair.leader.enable()
        pair.follower.enable()
        for role, arm in (("leader", pair.leader), ("follower", pair.follower)):
            try:
                self._wait_for_arm_role(pair.name, arm, "follower")
            except RuntimeError:
                log.warning(
                    "retrying follower mode before reset pair=%s role=%s arm=%s",
                    pair.name,
                    role,
                    arm.name,
                )
                arm.set_follower_mode()
                self._settle_after_role_switch(True)
                arm.enable()
                self._wait_for_arm_role(pair.name, arm, "follower")

    def teleop_step(self) -> tuple[int, dict[str, tuple[str, np.ndarray]]]:
        leader_states: list[ArmState] = []
        follower_states: list[ArmState] = []
        q_cmds: list[np.ndarray] = []

        for pair in self.pairs:
            leader_target = pair.leader.read_leader_joint_positions()
            if not _is_valid_joint_vector(leader_target):
                raise RuntimeError(f"Leader arm {pair.leader.name} did not provide a valid teleop target: {leader_target}")
            follower_before = pair.follower.read_state()
            follower_target = self._map_leader_to_follower(pair, leader_target)
            q_cmd = _limit_joint_step(
                follower_target,
                follower_before.q,
                self.config.teleop.command.joint_step_limit_rad,
            )
            self._command_follower(pair, q_cmd)

            leader_states.append(pair.leader.read_state())
            # Reuse the observation that produced q_cmd. An immediate second read makes
            # finite-difference acceleration use a sub-millisecond interval.
            follower_states.append(follower_before)
            q_cmds.append(q_cmd)

        timestamp_us = now_us()
        values = self._build_teleop_values(leader_states, follower_states, q_cmds)
        self._update_gripper_teleop(values)
        return timestamp_us, values

    def _command_follower(self, pair: ArmPairRuntime, q_cmd: np.ndarray) -> None:
        command = self.config.teleop.command
        if command.control_mode == "position":
            pair.follower.command_joint_positions(q_cmd)
            return
        if command.control_mode != "mit":
            raise RuntimeError(f"Unsupported control_mode={command.control_mode!r}")
        mit = command.mit
        pair.follower.command_joint_impedance(
            q_cmd,
            np.asarray(mit.v_des, dtype=np.float64),
            np.asarray(mit.kp, dtype=np.float64),
            np.asarray(mit.kd, dtype=np.float64),
            np.asarray(mit.t_ff, dtype=np.float64),
        )

    def _update_gripper_teleop(self, values: dict[str, tuple[str, np.ndarray]]) -> None:
        gripper = self.config.gripper
        if not gripper.enabled:
            return
        leader_values: list[np.ndarray] = []
        follower_values: list[np.ndarray] = []
        command_values: list[np.ndarray] = []
        now = time.monotonic()
        command_period = 1.0 / gripper.command_rate_hz
        for pair in self.pairs:
            read_leader = gripper.teleop_enabled or gripper.attach_to in {"leader", "both"}
            read_follower = gripper.teleop_enabled or gripper.attach_to in {"follower", "both"}
            leader_reader = getattr(
                pair.leader,
                "read_leader_gripper_state",
                pair.leader.read_gripper_state,
            )
            leader_state = leader_reader() if read_leader else None
            follower_state = pair.follower.read_gripper_state() if read_follower else None
            if leader_state is not None:
                previous_timestamp = self._leader_gripper_feedback_timestamp_us.get(pair.name)
                if previous_timestamp != leader_state.timestamp_us:
                    self._leader_gripper_feedback_timestamp_us[pair.name] = leader_state.timestamp_us
                    self._leader_gripper_feedback_change_t[pair.name] = now
                    self._leader_gripper_stale_warned.discard(pair.name)
                else:
                    last_change_t = self._leader_gripper_feedback_change_t.get(pair.name, now)
                    if now - last_change_t >= max(1.0, 2.0 * gripper.keepalive_s):
                        if pair.name not in self._leader_gripper_stale_warned:
                            log.warning(
                                "leader gripper feedback is stale pair=%s value=%.6f mode=%s; "
                                "no updated CAN gripper frame received",
                                pair.name,
                                leader_state.value,
                                leader_state.mode,
                            )
                            self._leader_gripper_stale_warned.add(pair.name)
            if leader_state is not None:
                leader_values.append(np.asarray([leader_state.value], dtype=np.float64))
            if follower_state is not None:
                follower_values.append(np.asarray([follower_state.value], dtype=np.float64))

            command_value = np.nan
            valid_leader_state = (
                leader_state is not None
                and np.isfinite(leader_state.value)
                and leader_state.mode == "width"
            )
            if gripper.teleop_enabled and valid_leader_state:
                command_value = float(
                    np.clip(
                        gripper.scale * leader_state.value + gripper.offset_m,
                        gripper.min_width_m,
                        gripper.max_width_m,
                    )
                )
                last_value = self._last_gripper_command.get(pair.name)
                last_mode = self._last_gripper_command_mode.get(pair.name)
                last_t = self._last_gripper_command_t.get(pair.name, float("-inf"))
                changed = (
                    last_value is None
                    or last_mode != "width"
                    or abs(command_value - last_value) >= gripper.deadband_m
                )
                due = now - last_t >= command_period
                keepalive_due = now - last_t >= gripper.keepalive_s
                if due and (changed or keepalive_due):
                    pair.follower.command_gripper(command_value, gripper.force_n, mode="width")
                    if pair.name not in self._gripper_command_announced:
                        log.info(
                            "gripper teleop active pair=%s leader=%.6fm command=%.6fm force=%.3fN",
                            pair.name,
                            leader_state.value,
                            command_value,
                            gripper.force_n,
                        )
                        self._gripper_command_announced.add(pair.name)
                    else:
                        log.debug(
                            "gripper command pair=%s leader=%.6fm command=%.6fm",
                            pair.name,
                            leader_state.value,
                            command_value,
                        )
                    self._last_gripper_command[pair.name] = command_value
                    self._last_gripper_command_mode[pair.name] = "width"
                    self._last_gripper_command_t[pair.name] = now
            elif gripper.teleop_enabled and pair.name not in self._gripper_feedback_warned:
                mode = leader_state.mode if leader_state is not None else "unavailable"
                log.warning(
                    "leader gripper feedback unavailable pair=%s mode=%s; command skipped",
                    pair.name,
                    mode,
                )
                self._gripper_feedback_warned.add(pair.name)
            command_values.append(np.asarray([command_value], dtype=np.float64))

        if follower_values:
            follower_value = _concat(follower_values)
            values["gripper_state"] = ("gripper", follower_value)
            values["gripper_value"] = ("gripper", follower_value)
            values["gripper_follower"] = ("gripper", follower_value)
        if leader_values:
            values["gripper_leader"] = ("gripper", _concat(leader_values))
        if command_values:
            values["gripper_cmd"] = ("gripper", _concat(command_values))

    def reset_to_rest(self) -> None:
        command = self.config.teleop.command
        log.info("resetting both arms to each pair's follower rest_q")
        self._hold_after_reset = True
        self._parked = False
        self._unrecorded_teleop = False
        self._teleop_reference.clear()
        for pair in self.pairs:
            self._prepare_pair_for_reset(pair)
        reset_targets = {
            pair.name: {
                "leader": pair.rest_q_follower.copy(),
                "follower": pair.rest_q_follower.copy(),
            }
            for pair in self.pairs
        }

        self._move_both_arms_to_reset_targets(reset_targets)
        deadline = time.monotonic() + command.reset_timeout_s

        while True:
            time.sleep(command.reset_wait_s)
            reset_errors = self._sample_reset_errors()
            max_error = max(
                float(np.max(np.abs(error)))
                for pair_errors in reset_errors.values()
                for error in pair_errors.values()
            )
            log.info(
                "dual-arm reset check from %d averaged samples: max joint error %.6f rad",
                max(command.reset_test_sample_time, 1),
                max_error,
            )
            if max_error <= command.reset_error_limit_rad:
                log.info("dual-arm reset self-check passed with limit %.6f rad", command.reset_error_limit_rad)
                self._set_parked_state()
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Dual-arm reset self-check failed: max joint error {max_error:.6f} rad "
                    f"> limit {command.reset_error_limit_rad:.6f} rad"
                )
            log.warning(
                "reset error %.6f rad exceeds limit %.6f rad; fine-tuning dual-arm reset targets",
                max_error,
                command.reset_error_limit_rad,
            )
            for pair in self.pairs:
                for role in ("leader", "follower"):
                    error = reset_errors[pair.name][role]
                    current_target = reset_targets[pair.name][role]
                    corrected_target = current_target + error
                    reset_targets[pair.name][role] = _limit_joint_step(
                        corrected_target,
                        current_target,
                        command.joint_step_limit_rad,
                    )
                    log.info(
                        "reset fine-tune pair=%s role=%s mean_error=%s target=%s",
                        pair.name,
                        role,
                        np.array2string(error, precision=6, suppress_small=True),
                        np.array2string(reset_targets[pair.name][role], precision=6, suppress_small=True),
                    )
            self._move_both_arms_to_reset_targets(reset_targets)

    def _move_both_arms_to_reset_targets(
        self,
        reset_targets: dict[str, dict[str, np.ndarray]],
    ) -> None:
        command = self.config.teleop.command
        timeout_s = command.reset_timeout_s
        start_q: dict[str, dict[str, np.ndarray]] = {}
        max_delta = 0.0
        for pair in self.pairs:
            start_q[pair.name] = {}
            for role, arm in (("leader", pair.leader), ("follower", pair.follower)):
                q = np.asarray(arm.read_state().q, dtype=np.float64).reshape(-1)
                target = reset_targets[pair.name][role]
                if not _is_valid_joint_vector(q) or q.size != target.size:
                    raise RuntimeError(f"Invalid {role} reset start joints from {arm.name}: {q}")
                start_q[pair.name][role] = q
                max_delta = max(max_delta, float(np.max(np.abs(target - q))))

        if not command.reset_interpolation_enabled or max_delta < 1e-9:
            steps = 1
        else:
            duration_s = max(command.reset_min_duration_s, max_delta / command.reset_joint_speed_rad_s)
            steps = max(
                1,
                math.ceil(duration_s * command.reset_interpolation_rate_hz),
                math.ceil(max_delta / command.reset_max_step_rad),
            )
        effective_duration_s = steps / command.reset_interpolation_rate_hz if steps > 1 else 0.0
        log.info(
            "dual-arm reset interpolation steps=%d duration=%.3fs max_delta=%.4frad",
            steps,
            effective_duration_s,
            max_delta,
        )

        interpolation_start = time.monotonic()
        for step_index in range(1, steps + 1):
            alpha = step_index / steps
            # Send both commands before sleeping so both arms advance together.
            for pair in self.pairs:
                for role, arm in (("leader", pair.leader), ("follower", pair.follower)):
                    start = start_q[pair.name][role]
                    target = reset_targets[pair.name][role]
                    arm.move_joints(start + alpha * (target - start))
            if steps > 1:
                next_t = interpolation_start + step_index / command.reset_interpolation_rate_hz
                remaining = next_t - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)

        timed_out: list[str] = []
        for pair in self.pairs:
            for role, arm in (("leader", pair.leader), ("follower", pair.follower)):
                if not arm.wait_motion_done(timeout_s):
                    timed_out.append(f"{pair.name}:{role}:{arm.name}")
        if timed_out:
            raise RuntimeError(f"Timed out waiting for reset arms: {timed_out}")

    def _set_parked_state(self) -> None:
        self._hold_after_reset = True
        self._parked = True
        self._unrecorded_teleop = False
        self._teleop_reference.clear()
        log.info("both arms parked in follower mode; waiting for r or t")

    def _sample_reset_errors(self) -> dict[str, dict[str, np.ndarray]]:
        command = self.config.teleop.command
        sample_count = max(command.reset_test_sample_time, 1)
        sample_period = 1.0 / max(command.idle_rate_hz, 1.0)
        samples: dict[str, dict[str, list[np.ndarray]]] = {
            pair.name: {"leader": [], "follower": []} for pair in self.pairs
        }
        for sample_index in range(sample_count):
            for pair in self.pairs:
                for role, arm in (("leader", pair.leader), ("follower", pair.follower)):
                    q = np.asarray(arm.read_state().q, dtype=np.float64).reshape(-1)
                    if not _is_valid_joint_vector(q) or q.size != pair.rest_q_follower.size:
                        raise RuntimeError(
                            f"{role.capitalize()} arm {arm.name} returned invalid reset joints: {q}"
                        )
                    samples[pair.name][role].append(q)
            if sample_index + 1 < sample_count:
                time.sleep(sample_period)
        return {
            pair.name: {
                role: pair.rest_q_follower - np.mean(samples[pair.name][role], axis=0)
                for role in ("leader", "follower")
            }
            for pair in self.pairs
        }

    def _map_leader_to_follower(self, pair: ArmPairRuntime, leader_q: np.ndarray) -> np.ndarray:
        mapping = self.config.teleop.command.teleop_mapping
        if mapping == "absolute":
            return leader_q
        if mapping != "relative_offset":
            raise RuntimeError(f"Unsupported teleop_mapping={mapping!r}")
        if pair.name not in self._teleop_reference:
            raise RuntimeError(f"Teleop reference for pair {pair.name} has not been initialized")
        leader_q0, follower_q0 = self._teleop_reference[pair.name]
        return follower_q0 + (leader_q - leader_q0)

    def _ensure_teleop_reference(self, pair: ArmPairRuntime) -> None:
        if pair.name in self._teleop_reference:
            return
        self._initialize_teleop_reference(pair)

    def _initialize_teleop_reference(self, pair: ArmPairRuntime) -> None:
        command = self.config.teleop.command
        follower_q0 = self._wait_for_valid_joints(
            lambda: pair.follower.read_state().q,
            command.input_ready_timeout_s,
            f"Follower arm {pair.follower.name}",
        )
        if command.pre_teleop_align_enabled:
            print(
                f"Move the leader arm near the follower pose. Teleoperation starts when "
                f"the max joint error is <= {command.pre_teleop_align_error_limit_rad:.4f} rad.",
                flush=True,
            )
            leader_q0 = self._wait_for_leader_alignment(pair, follower_q0)
        else:
            leader_q0 = self._wait_for_valid_joints(
                pair.leader.read_leader_joint_positions,
                command.input_ready_timeout_s,
                f"Leader arm {pair.leader.name}",
            )
        self._teleop_reference[pair.name] = (leader_q0.copy(), follower_q0.copy())
        log.info("initialized teleop reference pair=%s mapping=%s", pair.name, command.teleop_mapping)

    def _wait_for_leader_alignment(self, pair: ArmPairRuntime, target_q: np.ndarray) -> np.ndarray:
        command = self.config.teleop.command
        next_log_t = 0.0
        last_error = float("inf")
        last_q = np.empty((0,), dtype=np.float64)
        while True:
            last_q = np.asarray(pair.leader.read_leader_joint_positions(), dtype=np.float64).reshape(-1)
            if _is_valid_joint_vector(last_q):
                last_error = _max_abs_error(last_q, target_q)
                if last_error <= command.pre_teleop_align_error_limit_rad:
                    log.info("pre-teleop alignment passed pair=%s error=%.6f rad", pair.name, last_error)
                    return last_q
            now = time.monotonic()
            if now >= next_log_t:
                log.info(
                    "waiting for leader alignment pair=%s error=%.6f rad limit=%.6f rad",
                    pair.name,
                    last_error,
                    command.pre_teleop_align_error_limit_rad,
                )
                next_log_t = now + 1.0
            time.sleep(0.05)

    def _build_teleop_values(
        self,
        leader_states: list[ArmState],
        follower_states: list[ArmState],
        q_cmds: list[np.ndarray],
    ) -> dict[str, tuple[str, np.ndarray]]:
        values: dict[str, tuple[str, np.ndarray]] = {}
        states = self.config.robot_states

        if states.get("q") and states["q"].enabled:
            values["q_leader"] = ("q", _concat([state.q for state in leader_states]))
            values["q_follower"] = ("q", _concat([state.q for state in follower_states]))
            values["q_cmd"] = ("q", _concat(q_cmds))

        if states.get("velocity") and states["velocity"].enabled:
            values["dq_leader"] = ("velocity", _concat([state.dq for state in leader_states]))
            values["dq_follower"] = ("velocity", _concat([state.dq for state in follower_states]))

        if states.get("acceleration") and states["acceleration"].enabled:
            values["ddq_leader"] = ("acceleration", _concat([state.ddq for state in leader_states]))
            values["ddq_follower"] = ("acceleration", _concat([state.ddq for state in follower_states]))

        if states.get("ee_pose") and states["ee_pose"].enabled:
            values["ee_pose_leader"] = ("ee_pose", _pose_stack([state.ee_pose for state in leader_states]))
            values["ee_pose"] = ("ee_pose", _pose_stack([state.ee_pose for state in follower_states]))
            values["cmd_ee_pose"] = ("ee_pose", _pose_stack([state.ee_pose for state in leader_states]))

        if states.get("torque") and states["torque"].enabled:
            values["tau_leader"] = ("torque", _concat([state.torque for state in leader_states]))
            follower_tau = _concat([state.torque for state in follower_states])
            values["tau_follower"] = ("torque", follower_tau)

        if states.get("current") and states["current"].enabled:
            values["current_leader"] = ("current", _concat([state.current for state in leader_states]))
            values["current_follower"] = ("current", _concat([state.current for state in follower_states]))

        return values


def _rest_q(rest_q: tuple[float, ...]) -> np.ndarray:
    if rest_q:
        return np.asarray(rest_q, dtype=np.float64)
    return np.zeros(7, dtype=np.float64)


def _concat(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate([np.asarray(value, dtype=np.float64).reshape(-1) for value in values], axis=0)


def _pose_stack(poses: list[np.ndarray]) -> np.ndarray:
    if not poses:
        return np.empty((0, 4, 4), dtype=np.float64)
    normalized = [np.asarray(pose, dtype=np.float64).reshape(4, 4) for pose in poses]
    if len(normalized) == 1:
        return normalized[0]
    return np.stack(normalized, axis=0)


def _limit_joint_step(target_q: np.ndarray, current_q: np.ndarray, max_step_rad: float | None) -> np.ndarray:
    target = np.asarray(target_q, dtype=np.float64).reshape(-1)
    current = np.asarray(current_q, dtype=np.float64).reshape(-1)
    if not _is_valid_joint_vector(target):
        raise RuntimeError(f"Invalid target joint vector: {target}")
    if not _is_valid_joint_vector(current):
        return target
    if max_step_rad is None or current.size != target.size:
        return target
    delta = np.clip(target - current, -max_step_rad, max_step_rad)
    return current + delta


def _max_abs_error(actual_q: np.ndarray, target_q: np.ndarray) -> float:
    actual = np.asarray(actual_q, dtype=np.float64).reshape(-1)
    target = np.asarray(target_q, dtype=np.float64).reshape(-1)
    if actual.size != target.size:
        raise RuntimeError(f"Reset q size mismatch: actual={actual.size}, target={target.size}")
    return float(np.max(np.abs(actual - target)))


def _is_valid_joint_vector(q: np.ndarray) -> bool:
    array = np.asarray(q, dtype=np.float64).reshape(-1)
    return array.size > 0 and bool(np.isfinite(array).all())
