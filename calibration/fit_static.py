from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from calibration.common import aggregate_static_poses, load_dataset, load_plan
from calibration.static_model import StaticFitResult, TerminalStaticModel


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    plan = load_plan(args.config)
    dataset = load_dataset(args.data)
    pose_names, q, tau, tau_std = aggregate_static_poses(dataset)
    if len(pose_names) < plan.fit.min_poses:
        raise RuntimeError(
            f"need at least {plan.fit.min_poses} populated static poses; got {len(pose_names)}"
        )

    model = TerminalStaticModel(plan.model)
    result = model.fit(q, tau)
    reasons = _rejection_reasons(plan, result)
    report = _build_report(
        plan=plan,
        data_path=Path(args.data).expanduser().resolve(),
        dataset_metadata=dataset.metadata,
        pose_names=pose_names,
        tau_std=tau_std,
        model=model,
        result=result,
        reasons=reasons,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    _print_summary(output, result, reasons)
    return 0 if not reasons else 2


def _rejection_reasons(plan, result: StaticFitResult) -> list[str]:
    reasons: list[str] = []
    if result.regressor_rank < 4:
        reasons.append(f"terminal regressor rank is {result.regressor_rank}, expected 4")
    if result.regressor_condition_number > plan.fit.max_condition_number:
        reasons.append(
            "terminal regressor condition number "
            f"{result.regressor_condition_number:.6g} exceeds {plan.fit.max_condition_number:.6g}"
        )
    mass_min, mass_max = plan.fit.mass_bounds_kg
    if not mass_min <= result.terminal_mass_kg <= mass_max:
        reasons.append(
            f"terminal mass {result.terminal_mass_kg:.6g} kg is outside [{mass_min}, {mass_max}]"
        )
    if not np.isfinite(result.terminal_com_xyz_m).all() or np.any(
        np.abs(result.terminal_com_xyz_m) > plan.fit.max_abs_com_m
    ):
        reasons.append(
            "terminal COM is non-finite or exceeds "
            f"+/-{plan.fit.max_abs_com_m:.6g} m: {result.terminal_com_xyz_m.tolist()}"
        )
    if np.any(np.abs(result.joint_bias_nm) > plan.fit.max_abs_bias_nm):
        reasons.append(
            f"joint bias exceeds configured bounds: {result.joint_bias_nm.tolist()}"
        )
    return reasons


def _build_report(
    *,
    plan,
    data_path: Path,
    dataset_metadata: dict[str, Any],
    pose_names: tuple[str, ...],
    tau_std: np.ndarray,
    model: TerminalStaticModel,
    result: StaticFitResult,
    reasons: list[str],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "accepted": not reasons,
        "rejection_reasons": reasons,
        "input": {
            "dataset": str(data_path),
            "dataset_metadata": dataset_metadata,
            "pose_count": len(pose_names),
            "pose_names": list(pose_names),
            "median_tau_std_per_joint_nm": np.median(tau_std, axis=0).tolist(),
        },
        "model": {
            "urdf_path": str(plan.model.urdf_path),
            "locked_joint_names": list(plan.model.locked_joint_names),
            "terminal_joint_name": plan.model.terminal_joint_name,
            "gravity_m_s2": plan.model.gravity_m_s2.tolist(),
            "parameter_frame": f"{plan.model.terminal_joint_name} joint local frame",
            "parameter_order": ["mass", "mass_times_com_x", "mass_times_com_y", "mass_times_com_z"],
            "nominal_terminal_parameters": model.nominal_terminal_parameters.tolist(),
            "nominal_terminal_mass_kg": float(model.nominal_terminal_parameters[0]),
            "nominal_terminal_com_xyz_m": (
                model.nominal_terminal_parameters[1:4] / model.nominal_terminal_parameters[0]
            ).tolist(),
        },
        "fit": {
            "terminal_parameters": result.terminal_parameters.tolist(),
            "terminal_mass_kg": result.terminal_mass_kg,
            "terminal_com_xyz_m": result.terminal_com_xyz_m.tolist(),
            "joint_torque_bias_nm": result.joint_bias_nm.tolist(),
            "regressor_rank": result.regressor_rank,
            "regressor_condition_number": result.regressor_condition_number,
            "rmse_per_joint_nm": result.rmse_per_joint_nm.tolist(),
            "overall_rmse_nm": result.overall_rmse_nm,
        },
        "limits": {
            "max_condition_number": plan.fit.max_condition_number,
            "mass_bounds_kg": list(plan.fit.mass_bounds_kg),
            "max_abs_com_m": plan.fit.max_abs_com_m,
            "max_abs_bias_nm": plan.fit.max_abs_bias_nm.tolist(),
        },
        "urdf_written": False,
    }


def _print_summary(output: Path, result: StaticFitResult, reasons: list[str]) -> None:
    print(f"Static calibration report: {output}")
    print(f"accepted: {not reasons}")
    print(f"terminal mass: {result.terminal_mass_kg:.6f} kg")
    print("terminal COM in joint7 local frame: " + np.array2string(result.terminal_com_xyz_m, precision=6))
    print("joint torque bias [N.m]: " + np.array2string(result.joint_bias_nm, precision=6))
    print("RMSE per joint [N.m]: " + np.array2string(result.rmse_per_joint_nm, precision=6))
    print(
        f"regressor rank={result.regressor_rank}/4 "
        f"condition={result.regressor_condition_number:.6g}"
    )
    for reason in reasons:
        print(f"REJECTED: {reason}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit terminal aggregate mass/COM and seven joint torque biases from static Nero data."
    )
    parser.add_argument("--config", default="calibration/config.yaml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="calibration/results/static_fit.yaml")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())

