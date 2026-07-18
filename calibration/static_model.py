from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calibration.common import ModelSettings


@dataclass(frozen=True)
class StaticFitResult:
    terminal_parameters: np.ndarray
    terminal_mass_kg: float
    terminal_com_xyz_m: np.ndarray
    joint_bias_nm: np.ndarray
    predicted_tau_nm: np.ndarray
    residual_nm: np.ndarray
    rmse_per_joint_nm: np.ndarray
    overall_rmse_nm: float
    regressor_rank: int
    regressor_condition_number: float


class TerminalStaticModel:
    """Pinocchio static model with four identifiable terminal parameters."""

    def __init__(self, settings: ModelSettings) -> None:
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise RuntimeError("Static calibration requires Pinocchio 3.x") from exc
        self.pin = pin
        self.settings = settings
        full_model = pin.buildModelFromUrdf(str(settings.urdf_path))
        missing = [name for name in settings.locked_joint_names if not full_model.existJointName(name)]
        if missing:
            raise ValueError(f"locked joints not found in URDF: {missing}")
        locked_ids = [full_model.getJointId(name) for name in settings.locked_joint_names]
        self.model = (
            pin.buildReducedModel(full_model, locked_ids, pin.neutral(full_model))
            if locked_ids
            else full_model
        )
        if self.model.nq != 7 or self.model.nv != 7:
            raise ValueError(
                f"static calibration requires a reduced 7-DoF model; got nq={self.model.nq}, nv={self.model.nv}"
            )
        if not self.model.existJointName(settings.terminal_joint_name):
            raise ValueError(f"terminal joint not found in URDF: {settings.terminal_joint_name}")
        self.terminal_joint_id = self.model.getJointId(settings.terminal_joint_name)
        self.model.gravity.linear[:] = settings.gravity_m_s2
        self.data = self.model.createData()
        self.nominal_terminal_parameters = np.asarray(
            self.model.inertias[self.terminal_joint_id].toDynamicParameters(),
            dtype=np.float64,
        )[:4].copy()
        if self.nominal_terminal_parameters[0] <= 0:
            raise ValueError("nominal terminal aggregate mass must be positive")

    @property
    def lower_position_limit(self) -> np.ndarray:
        return np.asarray(self.model.lowerPositionLimit, dtype=np.float64).copy()

    @property
    def upper_position_limit(self) -> np.ndarray:
        return np.asarray(self.model.upperPositionLimit, dtype=np.float64).copy()

    def terminal_regression(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = _matrix("q", q, 7)
        regressors: list[np.ndarray] = []
        fixed_torques: list[np.ndarray] = []
        column_start = 10 * (self.terminal_joint_id - 1)
        zeros = np.zeros(7, dtype=np.float64)
        for pose in q:
            complete = np.asarray(
                self.pin.computeJointTorqueRegressor(self.model, self.data, pose, zeros, zeros),
                dtype=np.float64,
            )
            terminal = complete[:, column_start : column_start + 4].copy()
            nominal_tau = np.asarray(
                self.pin.rnea(self.model, self.data, pose, zeros, zeros),
                dtype=np.float64,
            ).copy()
            regressors.append(terminal)
            fixed_torques.append(nominal_tau - terminal @ self.nominal_terminal_parameters)
        return np.stack(regressors), np.stack(fixed_torques)

    def fit(self, q: np.ndarray, tau_measured: np.ndarray) -> StaticFitResult:
        q = _matrix("q", q, 7)
        tau_measured = _matrix("tau_measured", tau_measured, 7)
        if q.shape[0] != tau_measured.shape[0]:
            raise ValueError("q and tau_measured must have the same number of poses")
        regressors, fixed = self.terminal_regression(q)
        target = tau_measured - fixed

        # A constant bias exists independently on each joint. Centering per
        # joint removes those seven nuisance parameters before fitting the four
        # terminal mass/first-moment parameters.
        centered_regressor = regressors - np.mean(regressors, axis=0, keepdims=True)
        centered_target = target - np.mean(target, axis=0, keepdims=True)
        design = centered_regressor.reshape(-1, 4)
        observation = centered_target.reshape(-1)
        singular_values = np.linalg.svd(design, compute_uv=False)
        rank = int(np.linalg.matrix_rank(design))
        condition = (
            float(singular_values[0] / singular_values[-1])
            if singular_values.size and singular_values[-1] > np.finfo(np.float64).eps
            else float("inf")
        )
        terminal_parameters, _, _, _ = np.linalg.lstsq(design, observation, rcond=None)
        joint_bias = np.mean(target - regressors @ terminal_parameters, axis=0)
        predicted = fixed + regressors @ terminal_parameters + joint_bias
        residual = tau_measured - predicted
        mass = float(terminal_parameters[0])
        com = terminal_parameters[1:4] / mass if abs(mass) > 1e-12 else np.full(3, np.nan)
        return StaticFitResult(
            terminal_parameters=terminal_parameters,
            terminal_mass_kg=mass,
            terminal_com_xyz_m=np.asarray(com, dtype=np.float64),
            joint_bias_nm=joint_bias,
            predicted_tau_nm=predicted,
            residual_nm=residual,
            rmse_per_joint_nm=np.sqrt(np.mean(residual * residual, axis=0)),
            overall_rmse_nm=float(np.sqrt(np.mean(residual * residual))),
            regressor_rank=rank,
            regressor_condition_number=condition,
        )

    def predict(self, q: np.ndarray, terminal_parameters: np.ndarray, joint_bias_nm: np.ndarray) -> np.ndarray:
        terminal_parameters = _vector("terminal_parameters", terminal_parameters, 4)
        joint_bias_nm = _vector("joint_bias_nm", joint_bias_nm, 7)
        regressors, fixed = self.terminal_regression(q)
        return fixed + regressors @ terminal_parameters + joint_bias_nm

def _matrix(name: str, value: np.ndarray, width: int) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != width or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite (N, {width}) array")
    return matrix


def _vector(name: str, value: np.ndarray, size: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size != size or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite {size}D vector")
    return vector
