from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from nero_collection.config import ContactWrenchConfig, InverseDynamicsConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContactWrenchEstimate:
    wrench: np.ndarray
    tau_id: np.ndarray
    tau_residual: np.ndarray
    reconstruction_error: float
    condition_number: float


@dataclass(frozen=True)
class JointTorqueResidualEstimate:
    tau_id: np.ndarray
    tau_residual: np.ndarray


class PinocchioJointTorqueResidualEstimator:
    def __init__(self, config: InverseDynamicsConfig, dof: int = 7) -> None:
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise RuntimeError(
                "Realtime tau_ext plotting requires Pinocchio; install pin>=3,<4"
            ) from exc

        self.pin = pin
        self.config = config
        self.dof = int(dof)
        full_model = pin.buildModelFromUrdf(str(config.urdf_path))
        missing_joints = [
            name for name in config.locked_joint_names if not full_model.existJointName(name)
        ]
        if missing_joints:
            raise RuntimeError(f"Inverse-dynamics locked joints not found in URDF: {missing_joints}")
        locked_ids = [full_model.getJointId(name) for name in config.locked_joint_names]
        self.model = (
            pin.buildReducedModel(full_model, locked_ids, pin.neutral(full_model))
            if locked_ids
            else full_model
        )
        if self.model.nq != self.dof or self.model.nv != self.dof:
            raise RuntimeError(
                "Inverse-dynamics model must reduce to seven arm joints; "
                f"got nq={self.model.nq}, nv={self.model.nv}"
            )
        self.model.gravity.linear[:] = np.asarray(config.gravity_m_s2, dtype=np.float64)
        self.data = self.model.createData()
        log.info(
            "Pinocchio tau_ext estimator ready urdf=%s delay=%.3fs",
            config.urdf_path,
            config.delay_s,
        )

    def estimate(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
        tau_measured: np.ndarray,
    ) -> JointTorqueResidualEstimate:
        q = _finite_vector("q", q, self.dof)
        dq = _finite_vector("dq", dq, self.dof)
        ddq = _finite_vector("ddq", ddq, self.dof)
        tau_measured = _finite_vector("tau", tau_measured, self.dof)
        tau_id = np.asarray(
            self.pin.rnea(self.model, self.data, q, dq, ddq),
            dtype=np.float64,
        ).copy()
        return JointTorqueResidualEstimate(
            tau_id=tau_id,
            tau_residual=tau_id - tau_measured,
        )


class PinocchioContactWrenchEstimator:
    def __init__(self, config: ContactWrenchConfig, dof: int = 7) -> None:
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise RuntimeError(
                "Contact-wrench plotting requires Pinocchio; install pin>=3,<4"
            ) from exc

        self.pin = pin
        self.config = config
        self.dof = int(dof)
        full_model = pin.buildModelFromUrdf(str(config.urdf_path))
        missing_joints = [
            name for name in config.locked_joint_names if not full_model.existJointName(name)
        ]
        if missing_joints:
            raise RuntimeError(f"Contact-wrench locked joints not found in URDF: {missing_joints}")
        locked_ids = [full_model.getJointId(name) for name in config.locked_joint_names]
        self.model = (
            pin.buildReducedModel(full_model, locked_ids, pin.neutral(full_model))
            if locked_ids
            else full_model
        )
        if self.model.nq != self.dof or self.model.nv != self.dof:
            raise RuntimeError(
                "Contact-wrench model must reduce to seven arm joints; "
                f"got nq={self.model.nq}, nv={self.model.nv}"
            )
        self.model.gravity.linear[:] = np.asarray(config.gravity_m_s2, dtype=np.float64)
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId(config.frame_name)
        if self.frame_id == len(self.model.frames):
            raise RuntimeError(f"Contact-wrench frame not found in URDF: {config.frame_name}")
        self.reference_frame = (
            pin.ReferenceFrame.LOCAL
            if config.reference_frame == "local"
            else pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        log.info(
            "Pinocchio contact estimator ready urdf=%s frame=%s reference=%s delay=%.3fs damping=%.4g",
            config.urdf_path,
            config.frame_name,
            config.reference_frame,
            config.delay_s,
            config.damping,
        )

    def estimate(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
        tau_measured: np.ndarray,
    ) -> ContactWrenchEstimate:
        q = _finite_vector("q", q, self.dof)
        dq = _finite_vector("dq", dq, self.dof)
        ddq = _finite_vector("ddq", ddq, self.dof)
        tau_measured = _finite_vector("tau", tau_measured, self.dof)

        tau_id = np.asarray(
            self.pin.rnea(self.model, self.data, q, dq, ddq),
            dtype=np.float64,
        ).copy()
        tau_residual = tau_id - tau_measured
        self.pin.computeJointJacobians(self.model, self.data, q)
        self.pin.framesForwardKinematics(self.model, self.data, q)
        jacobian = np.asarray(
            self.pin.getFrameJacobian(
                self.model,
                self.data,
                self.frame_id,
                self.reference_frame,
            ),
            dtype=np.float64,
        )
        wrench_vector, error, condition = solve_damped_wrench(
            jacobian,
            tau_residual,
            self.config.damping,
        )
        spatial_force = self.pin.Force(wrench_vector)
        wrench = np.concatenate(
            (
                np.asarray(spatial_force.linear, dtype=np.float64).reshape(3),
                np.asarray(spatial_force.angular, dtype=np.float64).reshape(3),
            )
        )
        return ContactWrenchEstimate(
            wrench=wrench,
            tau_id=tau_id,
            tau_residual=tau_residual,
            reconstruction_error=error,
            condition_number=condition,
        )


def solve_damped_wrench(
    jacobian: np.ndarray,
    tau_residual: np.ndarray,
    damping: float,
) -> tuple[np.ndarray, float, float]:
    jacobian = np.asarray(jacobian, dtype=np.float64)
    tau_residual = np.asarray(tau_residual, dtype=np.float64).reshape(-1)
    if jacobian.shape[0] != 6 or jacobian.shape[1] != tau_residual.size:
        raise RuntimeError(
            "Expected a 6xN frame Jacobian and an N-dimensional torque residual; "
            f"got J={jacobian.shape}, tau={tau_residual.shape}"
        )
    if not np.isfinite(jacobian).all() or not np.isfinite(tau_residual).all():
        raise RuntimeError("Contact-wrench inputs must be finite")
    if not np.isfinite(damping) or damping <= 0:
        raise RuntimeError("Contact-wrench damping must be positive and finite")

    joint_to_wrench = jacobian.T
    u, singular_values, vt = np.linalg.svd(joint_to_wrench, full_matrices=False)
    gains = singular_values / (singular_values * singular_values + damping * damping)
    wrench = vt.T @ (gains * (u.T @ tau_residual))
    reconstructed = joint_to_wrench @ wrench
    denominator = max(float(np.linalg.norm(tau_residual)), 1e-9)
    reconstruction_error = float(np.linalg.norm(reconstructed - tau_residual) / denominator)
    smallest = float(np.min(singular_values)) if singular_values.size else 0.0
    condition_number = (
        float(np.max(singular_values) / smallest)
        if smallest > np.finfo(np.float64).eps
        else float("inf")
    )
    return wrench, reconstruction_error, condition_number


def _finite_vector(name: str, value: np.ndarray, size: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size != size or not np.isfinite(vector).all():
        raise RuntimeError(f"Contact-wrench estimator requires a finite {size}D {name}; got {vector}")
    return vector.copy()
