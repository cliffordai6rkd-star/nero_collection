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
    data_path = Path(args.data).expanduser().resolve()
    fit_path = Path(args.fit).expanduser().resolve()
    dataset = load_dataset(data_path)
    fit_report = yaml.safe_load(fit_path.read_text(encoding="utf-8")) or {}
    _verify_fit_report(fit_report, fit_path, data_path)

    pose_names, q_all, tau_all, tau_std = aggregate_static_poses(dataset)
    if len(pose_names) < plan.fit.min_poses:
        raise RuntimeError(
            f"need at least {plan.fit.min_poses} populated poses; got {len(pose_names)}"
        )
    model = TerminalStaticModel(plan.model)
    all_result = model.fit(q_all, tau_all)
    _verify_full_fit_parameters(fit_report, all_result)

    train_indices, holdout_indices = _split_pose_indices(
        len(pose_names),
        plan.fit.holdout_fraction,
        plan.fit.split_seed,
    )
    if train_indices.size < plan.fit.min_poses:
        raise RuntimeError(
            f"holdout split leaves {train_indices.size} train poses, below fit.min_poses={plan.fit.min_poses}"
        )
    train_result = model.fit(q_all[train_indices], tau_all[train_indices])
    nominal_bias = _fit_nominal_bias(
        model,
        q_all[train_indices],
        tau_all[train_indices],
    )
    nominal_holdout = model.predict(
        q_all[holdout_indices],
        model.nominal_terminal_parameters,
        nominal_bias,
    )
    calibrated_holdout = model.predict(
        q_all[holdout_indices],
        train_result.terminal_parameters,
        train_result.joint_bias_nm,
    )
    nominal_residual = tau_all[holdout_indices] - nominal_holdout
    calibrated_residual = tau_all[holdout_indices] - calibrated_holdout
    nominal_rmse_per_joint = _rmse(nominal_residual, axis=0)
    calibrated_rmse_per_joint = _rmse(calibrated_residual, axis=0)
    nominal_rmse = float(_rmse(nominal_residual))
    calibrated_rmse = float(_rmse(calibrated_residual))
    improvement_ratio = (
        (nominal_rmse - calibrated_rmse) / nominal_rmse
        if nominal_rmse > np.finfo(np.float64).eps
        else 0.0
    )

    round_ids = tuple(int(value) for value in np.unique(dataset.round_index))
    round_results = _fit_individual_rounds(dataset, pose_names, round_ids, model)
    stability = _round_stability(round_results)
    reasons = _rejection_reasons(
        plan=plan,
        train_result=train_result,
        round_ids=round_ids,
        improvement_ratio=improvement_ratio,
        stability=stability,
    )

    report = {
        "format_version": 1,
        "validation_type": "internal_pose_holdout",
        "created_at": datetime.now().astimezone().isoformat(),
        "accepted": not reasons,
        "rejection_reasons": reasons,
        "external_validation_performed": False,
        "input": {
            "fit_report": str(fit_path),
            "dataset": str(data_path),
            "dataset_metadata": dataset.metadata,
            "pose_count": len(pose_names),
            "round_ids": list(round_ids),
            "median_tau_std_per_joint_nm": np.median(tau_std, axis=0).tolist(),
        },
        "split": {
            "seed": plan.fit.split_seed,
            "holdout_fraction": plan.fit.holdout_fraction,
            "train_pose_names": [pose_names[index] for index in train_indices],
            "holdout_pose_names": [pose_names[index] for index in holdout_indices],
        },
        "holdout_metrics": {
            "nominal_bias_fitted_on_train_nm": nominal_bias.tolist(),
            "nominal_rmse_per_joint_nm": nominal_rmse_per_joint.tolist(),
            "nominal_overall_rmse_nm": nominal_rmse,
            "calibrated_rmse_per_joint_nm": calibrated_rmse_per_joint.tolist(),
            "calibrated_overall_rmse_nm": calibrated_rmse,
            "improvement_ratio": improvement_ratio,
            "train_regressor_rank": train_result.regressor_rank,
            "train_regressor_condition_number": train_result.regressor_condition_number,
        },
        "round_stability": {
            "mass_relative_range": stability["mass_relative_range"],
            "max_com_pair_distance_m": stability["max_com_pair_distance_m"],
            "joint_bias_range_nm": stability["joint_bias_range_nm"].tolist(),
            "round_fits": [
                {
                    "round_index": round_id,
                    "terminal_mass_kg": result.terminal_mass_kg,
                    "terminal_com_xyz_m": result.terminal_com_xyz_m.tolist(),
                    "joint_torque_bias_nm": result.joint_bias_nm.tolist(),
                    "overall_rmse_nm": result.overall_rmse_nm,
                    "regressor_rank": result.regressor_rank,
                    "regressor_condition_number": result.regressor_condition_number,
                }
                for round_id, result in zip(round_ids, round_results)
            ],
        },
        "limits": {
            "min_round_count": 3,
            "max_condition_number": plan.fit.max_condition_number,
            "min_holdout_improvement_ratio": plan.fit.min_holdout_improvement_ratio,
            "max_round_mass_relative_range": plan.fit.max_round_mass_relative_range,
            "max_round_com_spread_m": plan.fit.max_round_com_spread_m,
            "max_round_bias_range_nm": plan.fit.max_round_bias_range_nm.tolist(),
        },
        "urdf_written": False,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    _print_summary(output, report)
    return 0 if not reasons else 2


def _verify_fit_report(report: dict[str, Any], fit_path: Path, data_path: Path) -> None:
    if report.get("accepted") is not True:
        raise RuntimeError(f"fit report is not accepted: {fit_path}")
    fitted_dataset = report.get("input", {}).get("dataset")
    if fitted_dataset is None or Path(str(fitted_dataset)).expanduser().resolve() != data_path:
        raise RuntimeError("fit report was not generated from the selected calibration dataset")


def _verify_full_fit_parameters(report: dict[str, Any], result: StaticFitResult) -> None:
    fit = report.get("fit", {})
    terminal = np.asarray(fit.get("terminal_parameters"), dtype=np.float64).reshape(-1)
    bias = np.asarray(fit.get("joint_torque_bias_nm"), dtype=np.float64).reshape(-1)
    if terminal.size != 4 or bias.size != 7:
        raise ValueError("fit report contains invalid fitted parameters")
    if not np.allclose(terminal, result.terminal_parameters, rtol=1e-8, atol=1e-10):
        raise RuntimeError("fit report terminal parameters do not match the selected dataset")
    if not np.allclose(bias, result.joint_bias_nm, rtol=1e-8, atol=1e-10):
        raise RuntimeError("fit report torque biases do not match the selected dataset")


def _split_pose_indices(
    pose_count: int,
    holdout_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    shuffled = generator.permutation(pose_count)
    holdout_count = max(4, int(round(pose_count * holdout_fraction)))
    holdout_count = min(holdout_count, pose_count - 5)
    holdout = np.sort(shuffled[:holdout_count])
    train = np.sort(shuffled[holdout_count:])
    return train, holdout


def _fit_nominal_bias(
    model: TerminalStaticModel,
    q_train: np.ndarray,
    tau_train: np.ndarray,
) -> np.ndarray:
    zero_bias = np.zeros(7, dtype=np.float64)
    nominal = model.predict(q_train, model.nominal_terminal_parameters, zero_bias)
    return np.mean(tau_train - nominal, axis=0)


def _fit_individual_rounds(
    dataset,
    expected_pose_names: tuple[str, ...],
    round_ids: tuple[int, ...],
    model: TerminalStaticModel,
) -> tuple[StaticFitResult, ...]:
    results: list[StaticFitResult] = []
    for round_id in round_ids:
        names, q, tau, _ = aggregate_static_poses(dataset, round_index=round_id)
        if names != expected_pose_names:
            raise RuntimeError(
                f"round {round_id} does not contain the same populated poses as the full dataset"
            )
        results.append(model.fit(q, tau))
    return tuple(results)


def _round_stability(results: tuple[StaticFitResult, ...]) -> dict[str, Any]:
    if not results:
        raise RuntimeError("internal validation requires at least one round")
    masses = np.asarray([result.terminal_mass_kg for result in results], dtype=np.float64)
    coms = np.stack([result.terminal_com_xyz_m for result in results])
    biases = np.stack([result.joint_bias_nm for result in results])
    mean_mass = max(abs(float(np.mean(masses))), np.finfo(np.float64).eps)
    max_com_distance = 0.0
    for first in range(len(coms)):
        for second in range(first + 1, len(coms)):
            max_com_distance = max(
                max_com_distance,
                float(np.linalg.norm(coms[first] - coms[second])),
            )
    return {
        "mass_relative_range": float(np.ptp(masses) / mean_mass),
        "max_com_pair_distance_m": max_com_distance,
        "joint_bias_range_nm": np.ptp(biases, axis=0),
    }


def _rejection_reasons(
    *,
    plan,
    train_result: StaticFitResult,
    round_ids: tuple[int, ...],
    improvement_ratio: float,
    stability: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if len(round_ids) < 3:
        reasons.append(f"need at least 3 independent rounds; got {len(round_ids)}")
    if train_result.regressor_rank < 4:
        reasons.append(f"train terminal regressor rank is {train_result.regressor_rank}, expected 4")
    if train_result.regressor_condition_number > plan.fit.max_condition_number:
        reasons.append(
            f"train condition {train_result.regressor_condition_number:.6g} exceeds "
            f"{plan.fit.max_condition_number:.6g}"
        )
    if improvement_ratio < plan.fit.min_holdout_improvement_ratio:
        reasons.append(
            f"holdout RMSE improvement {improvement_ratio:.3%} is below "
            f"{plan.fit.min_holdout_improvement_ratio:.3%}"
        )
    if stability["mass_relative_range"] > plan.fit.max_round_mass_relative_range:
        reasons.append(
            f"round mass relative range {stability['mass_relative_range']:.3%} exceeds "
            f"{plan.fit.max_round_mass_relative_range:.3%}"
        )
    if stability["max_com_pair_distance_m"] > plan.fit.max_round_com_spread_m:
        reasons.append(
            f"round COM spread {stability['max_com_pair_distance_m']:.6g} m exceeds "
            f"{plan.fit.max_round_com_spread_m:.6g} m"
        )
    bias_range = np.asarray(stability["joint_bias_range_nm"])
    if np.any(bias_range > plan.fit.max_round_bias_range_nm):
        reasons.append(
            f"round joint-bias ranges exceed limits: measured={bias_range.tolist()} "
            f"limits={plan.fit.max_round_bias_range_nm.tolist()}"
        )
    return reasons


def _rmse(value: np.ndarray, axis=None) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return np.sqrt(np.mean(value * value, axis=axis))


def _print_summary(output: Path, report: dict[str, Any]) -> None:
    metrics = report["holdout_metrics"]
    stability = report["round_stability"]
    print(f"Internal validation report: {output}")
    print(f"accepted: {report['accepted']}")
    print(f"external_validation_performed: {report['external_validation_performed']}")
    print(
        f"holdout RMSE nominal={metrics['nominal_overall_rmse_nm']:.6f} N.m "
        f"calibrated={metrics['calibrated_overall_rmse_nm']:.6f} N.m "
        f"improvement={metrics['improvement_ratio']:.3%}"
    )
    print(
        f"round mass range={stability['mass_relative_range']:.3%} "
        f"COM spread={stability['max_com_pair_distance_m']:.6f} m"
    )
    for reason in report["rejection_reasons"]:
        print(f"REJECTED: {reason}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate static Nero calibration using pose holdout and repeated rounds."
    )
    parser.add_argument("--config", default="calibration/config.yaml")
    parser.add_argument("--fit", default="calibration/results/static_fit.yaml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="calibration/results/internal_validation.yaml")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
