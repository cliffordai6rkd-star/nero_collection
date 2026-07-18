from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from calibration.common import load_plan
from calibration.static_model import TerminalStaticModel


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    plan = load_plan(args.config)
    fit_path = Path(args.fit).expanduser().resolve()
    report = yaml.safe_load(fit_path.read_text(encoding="utf-8")) or {}
    if report.get("accepted") is not True:
        raise RuntimeError("refusing to write URDF from a fit report that is not accepted")
    validation_path = Path(args.validation).expanduser().resolve()
    validation = yaml.safe_load(validation_path.read_text(encoding="utf-8")) or {}
    if validation.get("accepted") is not True:
        raise RuntimeError("refusing to write URDF without an accepted internal validation")
    if validation.get("validation_type") != "internal_pose_holdout":
        raise RuntimeError("URDF writer requires an internal pose-holdout validation report")
    if validation.get("external_validation_performed") is not False:
        raise RuntimeError("internal validation report must mark external_validation_performed=false")
    validated_fit = validation.get("input", {}).get("fit_report")
    if validated_fit is None or Path(str(validated_fit)).expanduser().resolve() != fit_path:
        raise RuntimeError("internal validation was not generated from the selected fit report")
    fit = report.get("fit", {})
    terminal_parameters = np.asarray(fit.get("terminal_parameters"), dtype=np.float64).reshape(-1)
    joint_bias = np.asarray(fit.get("joint_torque_bias_nm"), dtype=np.float64).reshape(-1)
    if terminal_parameters.size != 4 or not np.isfinite(terminal_parameters).all():
        raise ValueError("fit report has invalid terminal parameters")
    if joint_bias.size != 7 or not np.isfinite(joint_bias).all():
        raise ValueError("fit report has invalid joint torque biases")

    model = TerminalStaticModel(plan.model)
    mass_kg = float(terminal_parameters[0])
    if mass_kg <= 0:
        raise RuntimeError("refusing to write a non-positive terminal mass")
    com_xyz_m = terminal_parameters[1:4] / mass_kg
    nominal_inertia = model.model.inertias[model.terminal_joint_id]
    inertia_scale = mass_kg / float(nominal_inertia.mass)
    inertia_com = np.asarray(nominal_inertia.inertia, dtype=np.float64) * inertia_scale
    _validate_inertia(inertia_com)

    source = plan.model.urdf_path.resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else source.with_name(f"{source.stem}_calibrated_static{source.suffix}")
    )
    if output == source:
        raise RuntimeError("refusing to overwrite the source URDF")
    if output.parent != source.parent:
        raise RuntimeError(
            "calibrated URDF must stay beside the source URDF so relative mesh paths remain valid"
        )

    tree = ET.parse(source)
    root = tree.getroot()
    target_link_name = _joint_child_link(root, plan.model.terminal_joint_name)
    descendant_links = _descendant_links(root, target_link_name)
    for link_name in descendant_links:
        link = _find_link(root, link_name)
        inertial = link.find("inertial")
        if inertial is not None:
            link.remove(inertial)
    _set_inertial(
        _find_link(root, target_link_name),
        mass_kg=mass_kg,
        com_xyz_m=com_xyz_m,
        inertia_com=inertia_com,
    )
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)

    # Confirm the generated file still loads as the same reduced 7-DoF model.
    generated_settings = type(plan.model)(
        urdf_path=output,
        locked_joint_names=plan.model.locked_joint_names,
        terminal_joint_name=plan.model.terminal_joint_name,
        gravity_m_s2=plan.model.gravity_m_s2,
    )
    generated_model = TerminalStaticModel(generated_settings)
    generated_parameters = generated_model.nominal_terminal_parameters
    if not np.allclose(generated_parameters, terminal_parameters, rtol=1e-7, atol=1e-9):
        raise RuntimeError(
            "generated URDF terminal parameters do not match the fit: "
            f"generated={generated_parameters}, fitted={terminal_parameters}"
        )

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "source_urdf": str(source),
        "generated_urdf": str(output),
        "fit_report": str(fit_path),
        "internal_validation": str(validation_path),
        "external_validation_performed": False,
        "target_link": target_link_name,
        "collapsed_links": sorted(descendant_links),
        "terminal_mass_kg": mass_kg,
        "terminal_com_xyz_m": com_xyz_m.tolist(),
        "terminal_inertia_com_kg_m2": inertia_com.tolist(),
        "joint_torque_bias_nm": joint_bias.tolist(),
        "assumptions": [
            "arm kinematic geometry, joint axes, signs, and zero positions are correct",
            "gripper joints are locked at the Pinocchio neutral configuration",
            "terminal rotational inertia uses the nominal aggregate inertia scaled by fitted mass",
            "joint torque biases are not representable in URDF and remain in this manifest",
        ],
    }
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print(f"Generated calibrated URDF: {output}")
    print(f"Generated calibration manifest: {manifest_path}")
    print("The source URDF was not modified.")
    return 0


def _joint_child_link(root: ET.Element, joint_name: str) -> str:
    for joint in root.findall("joint"):
        if joint.get("name") == joint_name:
            child = joint.find("child")
            if child is None or not child.get("link"):
                break
            return str(child.get("link"))
    raise ValueError(f"joint {joint_name!r} or its child link was not found in URDF")


def _descendant_links(root: ET.Element, start_link: str) -> set[str]:
    children: dict[str, list[str]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_name = parent.get("link")
        child_name = child.get("link")
        if parent_name and child_name:
            children.setdefault(parent_name, []).append(child_name)
    result = {start_link}
    pending = [start_link]
    while pending:
        parent = pending.pop()
        for child in children.get(parent, []):
            if child not in result:
                result.add(child)
                pending.append(child)
    return result


def _find_link(root: ET.Element, link_name: str) -> ET.Element:
    for link in root.findall("link"):
        if link.get("name") == link_name:
            return link
    raise ValueError(f"link {link_name!r} not found in URDF")


def _set_inertial(
    link: ET.Element,
    *,
    mass_kg: float,
    com_xyz_m: np.ndarray,
    inertia_com: np.ndarray,
) -> None:
    inertial = ET.Element("inertial")
    ET.SubElement(
        inertial,
        "origin",
        {"rpy": "0 0 0", "xyz": " ".join(_format(value) for value in com_xyz_m)},
    )
    ET.SubElement(inertial, "mass", {"value": _format(mass_kg)})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": _format(inertia_com[0, 0]),
            "ixy": _format(inertia_com[0, 1]),
            "ixz": _format(inertia_com[0, 2]),
            "iyy": _format(inertia_com[1, 1]),
            "iyz": _format(inertia_com[1, 2]),
            "izz": _format(inertia_com[2, 2]),
        },
    )
    link.insert(0, inertial)


def _validate_inertia(inertia: np.ndarray) -> None:
    if inertia.shape != (3, 3) or not np.isfinite(inertia).all():
        raise ValueError("terminal inertia must be a finite 3x3 matrix")
    eigenvalues = np.linalg.eigvalsh(inertia)
    if np.any(eigenvalues <= 0):
        raise ValueError(f"terminal inertia is not positive definite: eigenvalues={eigenvalues}")
    principal = np.sort(eigenvalues)
    if principal[2] > principal[0] + principal[1] + 1e-12:
        raise ValueError(f"terminal inertia violates the triangle inequality: {principal}")


def _format(value: float) -> str:
    return f"{float(value):.12g}"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a new static-calibrated URDF without modifying the source URDF."
    )
    parser.add_argument("--config", default="calibration/config.yaml")
    parser.add_argument("--fit", default="calibration/results/static_fit.yaml")
    parser.add_argument(
        "--validation",
        default="calibration/results/internal_validation.yaml",
    )
    parser.add_argument("--output")
    parser.add_argument("--manifest", default="calibration/results/calibrated_urdf_manifest.yaml")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
