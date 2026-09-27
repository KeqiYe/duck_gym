#!/usr/bin/env python3
"""Independent CPU analysis of aligned native/Newton long AVBD trajectories.

Relative position = ||p_native-p_newton||_2 / ||p_newton||_2.
Relative height = |z_native-z_newton| / |z_newton|.
A zero denominator is undefined, including 0/0; no epsilon is added.
No physical acceptance thresholds are imposed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Mapping, Sequence

import numpy as np

FIELDS = {"com_position": 3, "orientation": 4, "linear_velocity": 3, "angular_velocity": 3}
CHECKPOINTS = (.2, 1.0, 5.0, 10.0)


def _norm(x, axis=-1):
    """Scaled Euclidean norm, avoiding false zero for tiny nonzero vectors."""
    x = np.asarray(x, dtype=np.float64)
    scale = np.max(np.abs(x), axis=axis, keepdims=True)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scaled = np.divide(x, scale, out=np.zeros_like(x), where=scale != 0)
        return np.squeeze(scale, axis=axis) * np.sqrt(np.sum(scaled * scaled, axis=axis))


def _number(x):
    value = float(x)
    return value if np.isfinite(value) else None


def _safe_json(value):
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _safe_json(value.tolist())
    if isinstance(value, np.generic):
        return _safe_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _ratio(numerator, denominator, finite_input):
    defined = finite_input & np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
    value = np.full(numerator.shape, np.nan, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        np.divide(numerator, denominator, out=value, where=defined)
    reasons = {
        "zero_reference_denominator": finite_input & (denominator == 0),
        "nonfinite_input": ~finite_input,
        "nonfinite_computed_norm": finite_input & (~np.isfinite(numerator) | ~np.isfinite(denominator)),
    }
    return value, reasons


def _validate(native, newton, names, root):
    if not names or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("body_names must be nonempty unique strings")
    if root not in names:
        raise ValueError("root_body_name must occur exactly once in body_names")
    for label, data in (("native", native), ("newton", newton)):
        missing = {"time", *FIELDS} - data.keys()
        if missing:
            raise ValueError(f"{label} missing arrays: {sorted(missing)}")
        t = np.asarray(data["time"])
        if t.ndim != 1 or not len(t) or not np.isfinite(t).all() or not np.all(np.diff(t) > 0):
            raise ValueError(f"{label} time must be finite, nonempty, one-dimensional and strictly increasing")
        shape = np.asarray(data["com_position"]).shape
        if len(shape) != 4 or shape[0] != len(t) or shape[1] < 1 or shape[2] != len(names) or shape[3] != 3:
            raise ValueError(f"{label} com_position must have shape [time, env, body_names, 3]")
        for key, width in FIELDS.items():
            x = np.asarray(data[key])
            if x.shape != (*shape[:3], width) or x.dtype.kind != "f":
                raise ValueError(f"{label} {key} must be a floating array with shape {(*shape[:3], width)}")
    if not np.array_equal(native["time"], newton["time"]):
        raise ValueError("Native/Newton time arrays must match exactly; no interpolation or resampling is performed")
    if native["com_position"].shape != newton["com_position"].shape:
        raise ValueError("Native/Newton environment and body axes must match")


def analyze_trajectories(native: Mapping[str, np.ndarray], newton: Mapping[str, np.ndarray],
                         body_names: Sequence[str], root_body_name: str):
    """Return (JSON-compatible report, per-frame NumPy arrays), without file I/O."""
    names = list(body_names)
    _validate(native, newton, names, root_body_name)
    time = np.asarray(native["time"], dtype=np.float64)
    root_index = names.index(root_body_name)
    a = {key: np.asarray(native[key], dtype=np.float64) for key in FIELDS}
    b = {key: np.asarray(newton[key], dtype=np.float64) for key in FIELDS}
    finite = {key: np.isfinite(a[key]).all(-1) & np.isfinite(b[key]).all(-1) for key in FIELDS}
    height_finite = np.isfinite(a["com_position"][..., 2]) & np.isfinite(b["com_position"][..., 2])
    with np.errstate(over="ignore", invalid="ignore"):
        difference = a["com_position"] - b["com_position"]
        pos_abs = np.where(finite["com_position"], _norm(difference), np.nan)
        height_abs = np.where(height_finite, np.abs(difference[..., 2]), np.nan)
        pos_den = _norm(b["com_position"])
        height_den = np.abs(b["com_position"][..., 2])
        pos_rel, pos_reasons = _ratio(pos_abs, pos_den, finite["com_position"])
        height_rel, height_reasons = _ratio(height_abs, height_den, height_finite)
        linear_abs = np.where(finite["linear_velocity"], _norm(a["linear_velocity"] - b["linear_velocity"]), np.nan)
        angular_abs = np.where(finite["angular_velocity"], _norm(a["angular_velocity"] - b["angular_velocity"]), np.nan)
    norms_a, norms_b = _norm(a["orientation"]), _norm(b["orientation"])
    quat_valid = (finite["orientation"] & np.isfinite(norms_a) & np.isfinite(norms_b)
                  & (norms_a > 0) & (norms_b > 0))
    qa = np.divide(a["orientation"], norms_a[..., None], out=np.zeros_like(a["orientation"]),
                   where=quat_valid[..., None])
    qb = np.divide(b["orientation"], norms_b[..., None], out=np.zeros_like(b["orientation"]),
                   where=quat_valid[..., None])
    # Sign-invariant shortest angle. The difference/sum form remains stable near
    # zero angle, unlike acos(abs(dot(q_a,q_b))).
    qb = qb * np.where(np.sum(qa * qb, axis=-1) < 0, -1.0, 1.0)[..., None]
    orientation_abs = np.where(quat_valid, 4 * np.arctan2(_norm(qa - qb), _norm(qa + qb)), np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        frame_env_num = _norm(difference, axis=(2, 3))
        frame_env_den = _norm(b["com_position"], axis=(2, 3))
        frame_num = _norm(difference, axis=(1, 2, 3))
        frame_den = _norm(b["com_position"], axis=(1, 2, 3))
    frame_env_rel, frame_env_reasons = _ratio(frame_env_num, frame_env_den, finite["com_position"].all(2))
    frame_rel, frame_reasons = _ratio(frame_num, frame_den, finite["com_position"].all((1, 2)))

    metrics = {
        "position_relative": (pos_rel, pos_abs, pos_den, "ratio", pos_reasons, "com_position"),
        "height_relative": (height_rel, height_abs, height_den, "ratio", height_reasons, "height"),
        "position_absolute_m": (pos_abs, pos_abs, None, "m", {"nonfinite_input": ~finite["com_position"]}, "com_position"),
        "height_absolute_m": (height_abs, height_abs, None, "m", {"nonfinite_input": ~height_finite}, "height"),
        "orientation_absolute_rad": (orientation_abs, orientation_abs, None, "rad",
            {"nonfinite_input": ~finite["orientation"], "zero_quaternion": finite["orientation"] & ((norms_a == 0) | (norms_b == 0)),
             "nonfinite_quaternion_norm": finite["orientation"] & (~np.isfinite(norms_a) | ~np.isfinite(norms_b))}, "orientation"),
        "linear_velocity_absolute_m_s": (linear_abs, linear_abs, None, "m/s",
            {"nonfinite_input": ~finite["linear_velocity"]}, "linear_velocity"),
        "angular_velocity_absolute_rad_s": (angular_abs, angular_abs, None, "rad/s",
            {"nonfinite_input": ~finite["angular_velocity"]}, "angular_velocity"),
    }

    def summarize(spec, last=None, body=None):
        values, absolute, denominator, unit, reasons, field = spec
        stop = len(time) if last is None else last + 1
        select = (slice(0, stop), slice(None), slice(None) if body is None else slice(body, body + 1))
        x = values[select]
        defined = ~np.isnan(x)
        result = dict(total_count=int(x.size), defined_count=int(defined.sum()),
                      finite_count=int(np.isfinite(x).sum()), undefined_count=int(np.isnan(x).sum()),
                      positive_infinite_count=int(np.isposinf(x).sum()), unit=unit,
                      undefined_reasons={k: int(v[select].sum()) for k, v in reasons.items()}, maximum=None)
        if defined.any():
            local = np.unravel_index(np.argmax(np.where(defined, x, -np.inf)), x.shape)
            idx = (int(local[0]), int(local[1]), int(local[2] if body is None else body))
            value = values[idx]
            event = dict(value=_number(value), numeric_status="finite" if np.isfinite(value) else "positive_infinity",
                         frame_index=idx[0], time_s=float(time[idx[0]]), env_index=idx[1],
                         body_index=idx[2], body_name=names[idx[2]], absolute_error=_number(absolute[idx]))
            if denominator is not None:
                with np.errstate(over="ignore", invalid="ignore"):
                    percent = value * 100.0
                event.update(reference_denominator=_number(denominator[idx]), percent=_number(percent),
                             percent_numeric_status="finite" if np.isfinite(percent) else "positive_infinity")
            if field == "height":
                event.update(native_value=float(a["com_position"][idx][2]), newton_value=float(b["com_position"][idx][2]))
            else:
                event.update(native_value=a[field][idx].tolist(), newton_value=b[field][idx].tolist())
            result["maximum"] = event
        return result

    def frame_summary(values, numerator, denominator, reasons, last=None):
        stop = len(time) if last is None else last + 1
        x = values[:stop]; defined = ~np.isnan(x)
        result = dict(total_count=int(x.size), defined_count=int(defined.sum()),
                      undefined_count=int(np.isnan(x).sum()), positive_infinite_count=int(np.isposinf(x).sum()),
                      undefined_reasons={k: int(v[:stop].sum()) for k, v in reasons.items()}, maximum=None)
        if defined.any():
            idx = tuple(int(v) for v in np.unravel_index(np.argmax(np.where(defined, x, -np.inf)), x.shape))
            with np.errstate(over="ignore", invalid="ignore"):
                percent = values[idx] * 100.0
            result["maximum"] = dict(value=_number(values[idx]), percent=_number(percent),
                numeric_status="finite" if np.isfinite(values[idx]) else "positive_infinity",
                percent_numeric_status="finite" if np.isfinite(percent) else "positive_infinity",
                frame_index=idx[0], time_s=float(time[idx[0]]),
                absolute_l2_error_m=_number(numerator[idx]), reference_l2_norm_m=_number(denominator[idx]))
            if len(idx) == 2:
                result["maximum"]["env_index"] = idx[1]
        return result

    def frame_group(last=None):
        return {
            "per_frame_per_env_all_bodies": frame_summary(frame_env_rel, frame_env_num, frame_env_den, frame_env_reasons, last),
            "per_frame_all_envs_all_bodies": frame_summary(frame_rel, frame_num, frame_den, frame_reasons, last),
        }

    cumulative = []
    for checkpoint in CHECKPOINTS:
        indexes = np.flatnonzero(time == checkpoint)
        if not len(indexes):
            cumulative.append(dict(requested_time_s=checkpoint, available=False,
                                   reason="No exactly matching recorded timestamp; no interpolation"))
            continue
        idx = int(indexes[0])
        cumulative.append(dict(requested_time_s=checkpoint, available=True, through_frame_index=idx,
            sample_time_s=float(time[idx]), inclusive_of_initial_frame=True,
            metrics={k: summarize(v, last=idx) for k, v in metrics.items()},
            root_body={k: summarize(v, last=idx, body=root_index) for k, v in metrics.items()},
            position_l2_normalized=frame_group(idx)))

    input_info = {}
    for label, source, norms in (("native", native, norms_a), ("newton", newton, norms_b)):
        good_norm = np.isfinite(norms)
        input_info[label] = {
            "arrays": {key: dict(shape=list(np.asarray(source[key]).shape), dtype=str(np.asarray(source[key]).dtype),
                                nonfinite_scalar_count=int((~np.isfinite(source[key])).sum())) for key in ("time", *FIELDS)},
            "quaternion_zero_norm_count": int((norms == 0).sum()),
            "quaternion_max_abs_norm_minus_one": float(np.max(np.abs(norms[good_norm] - 1))) if good_norm.any() else None,
        }
    report = dict(analysis_complete=True, physical_acceptance_thresholds=None,
        reference="newton", frames=len(time), num_envs=int(a["com_position"].shape[1]),
        body_names=names, root_body_name=root_body_name, root_body_index=root_index,
        time_start_s=float(time[0]), time_end_s=float(time[-1]),
        definitions={
            "position_relative": "||p_native-p_newton||2 / ||p_newton||2, where p is world COM position in meters",
            "height_relative": "|z_native-z_newton| / |z_newton|; signed heights use an absolute denominator",
            "zero_denominator": "Undefined even for 0/0; no epsilon is added",
            "undefined_storage": "JSON maximum=null if no defined samples; NPZ per-sample undefined=NaN; counts/reasons retained",
            "infinite_ratio": "A positive overflow remains a maximum with numeric_status=positive_infinity and JSON value=null",
            "position_l2_per_env": "At each frame/env: sqrt(sum_body,xyz (p_native-p_newton)^2) / sqrt(sum_body,xyz p_newton^2)",
            "position_l2_all_envs": "At each frame: same norm including every env and body; never omits invalid bodies",
            "orientation": "Sign-invariant shortest rotation angle [0,pi], both input quaternions normalized first; common principal-inertia frame, wxyz",
            "velocity": "Euclidean absolute difference, world COM linear velocity m/s and world angular velocity rad/s",
            "tie_rule": "First maximum in frame/env/body C order; indexes are zero-based",
            "coordinates": "Position percentages depend on the chosen world origin; height percentages grow near z_reference=0",
            "maxima": "Each metric has its own maximum event; do not combine numerator/denominator from different events",
            "validity": "Nonfinite input and zero quaternions are explicitly undefined; no numerical threshold determines physical success",
        },
        inputs=input_info, global_metrics={k: summarize(v) for k, v in metrics.items()},
        per_body=[dict(body_index=i, body_name=name, metrics={k: summarize(v, body=i) for k, v in metrics.items()})
                  for i, name in enumerate(names)],
        root_body_metrics={k: summarize(v, body=root_index) for k, v in metrics.items()},
        position_l2_normalized=frame_group(), cumulative_checkpoints=cumulative)
    arrays = dict(time=time, body_names=np.asarray(names), position_absolute_m=pos_abs,
        height_absolute_m=height_abs, position_relative=pos_rel, height_relative=height_rel,
        position_reference_norm_m=pos_den, height_reference_abs_m=height_den,
        orientation_absolute_rad=orientation_abs, linear_velocity_absolute_m_s=linear_abs,
        angular_velocity_absolute_rad_s=angular_abs,
        position_l2_relative_per_frame_env=frame_env_rel, position_l2_error_per_frame_env_m=frame_env_num,
        position_l2_reference_per_frame_env_m=frame_env_den,
        position_l2_relative_per_frame_all_envs=frame_rel, position_l2_error_per_frame_all_envs_m=frame_num,
        position_l2_reference_per_frame_all_envs_m=frame_den)
    return _safe_json(report), arrays


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def _read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in ("time", *FIELDS)}


def _write_readme(path, report):
    def value(metric, relative=False):
        event = metric["maximum"]
        if event is None:
            return "未定义"
        x = event.get("percent") if relative else event["value"]
        return (f"{x:.8g}{'%' if relative else ''}" if x is not None else
                event.get("percent_numeric_status", event["numeric_status"]) if relative else event["numeric_status"])
    rows = ["# AVBD 长轨迹误差（Newton 为参考）", "",
        f"记录 {report['frames']} 帧、{report['num_envs']} env、{len(report['body_names'])} 个刚体，"
        f"时间 {report['time_start_s']:.6g}–{report['time_end_s']:.6g} s。没有新增物理验收阈值。", "",
        "| 指标 | 全刚体最大值 | 根刚体最大值 |", "| --- | ---: | ---: |"]
    for key, name in (("position_relative", "COM位置相对误差"), ("height_relative", "高度相对误差"),
                      ("position_absolute_m", "COM位置绝对差 / m"), ("height_absolute_m", "高度绝对差 / m"),
                      ("orientation_absolute_rad", "姿态短角 / rad"), ("linear_velocity_absolute_m_s", "COM线速度差 / m/s"),
                      ("angular_velocity_absolute_rad_s", "角速度差 / rad/s")):
        relative = key.endswith("_relative")
        rows.append(f"| {name} | {value(report['global_metrics'][key], relative)} | {value(report['root_body_metrics'][key], relative)} |")
    rows += ["", "相对误差最大事件：", ""]
    for key in ("position_relative", "height_relative"):
        m = report["global_metrics"][key]; e = m["maximum"]
        if e:
            rows.append(f"- {key}：frame={e['frame_index']}，t={e['time_s']:.9g} s，env={e['env_index']}，"
                        f"body={e['body_name']}；绝对差={e['absolute_error']:.9g} m，参考分母={e['reference_denominator']:.9g} m。"
                        f"分母为零的样本数={m['undefined_reasons']['zero_reference_denominator']}。")
        else:
            rows.append(f"- {key}：所有样本未定义。")
    rows += ["", "| 截至时间 s | 最大位置相对误差 | 最大高度相对误差 | 全env/全body逐帧L2归一化最大值 |",
             "| --- | ---: | ---: | ---: |"]
    for c in report["cumulative_checkpoints"]:
        if c["available"]:
            rows.append(f"| {c['sample_time_s']:g} | {value(c['metrics']['position_relative'], True)} | "
                        f"{value(c['metrics']['height_relative'], True)} | "
                        f"{value(c['position_l2_normalized']['per_frame_all_envs_all_bodies'], True)} |")
        else:
            rows.append(f"| {c['requested_time_s']:g} | 无对应记录 | — | — |")
    rows += ["", "分母为零时明确未定义，未加入 epsilon。位置百分比依赖世界原点，高度百分比可能被近零参考高度放大；"
             "最大事件、绝对差、逐刚体摘要和逐帧L2归一化误差应一起阅读。不同指标的最大值通常来自不同事件。",
             "", "姿态采用已统一主惯性坐标系的 wxyz 四元数，先归一化，再计算符号不敏感短角。"
             "非有限输入/零四元数的数量保留在报告中；分析完成不表示轨迹通过物理验收。",
             "", "完整统计见 report.json；逐帧/逐env/逐body误差见 per-frame-errors.npz。"]
    path.write_text("\n".join(rows) + "\n")


def _self_test():
    t = np.array([0., .2, 1., 5., 10.])
    p = np.broadcast_to([1., 2., 2.], (5, 2, 2, 3)).copy()
    q = np.zeros((5, 2, 2, 4)); q[..., 0] = 1
    v = np.zeros_like(p)
    ref = dict(time=t, com_position=p, orientation=q, linear_velocity=v.copy(), angular_velocity=v.copy())
    native = {k: x.copy() for k, x in ref.items()}
    ref["com_position"][0, 0] = 0; native["com_position"][0, 0] = 0
    native["com_position"][0, 0, 1, 0] = 1
    ref["com_position"][1, 0, 0, 2] = 0; native["com_position"][1, 0, 0, 2] = 1
    native["com_position"][4, 1, 1, 0] += 3
    native["orientation"] *= -1
    native["orientation"][3, 1, 1] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
    native["linear_velocity"][2, 0, 1] = [0, 3, 4]
    native["angular_velocity"][3, 0, 0] = [0, 0, 2]
    report, arrays = analyze_trajectories(native, ref, ["root", "foot"], "root")
    e = report["global_metrics"]["position_relative"]["maximum"]
    assert e["value"] == 1 and e["percent"] == 100 and (e["frame_index"], e["env_index"], e["body_name"]) == (4, 1, "foot")
    assert e["reference_denominator"] == 3 and e["absolute_error"] == 3
    assert report["global_metrics"]["position_relative"]["undefined_reasons"]["zero_reference_denominator"] == 2
    assert report["global_metrics"]["height_relative"]["undefined_reasons"]["zero_reference_denominator"] == 3
    assert np.isnan(arrays["position_l2_relative_per_frame_env"][0, 0])
    np.testing.assert_allclose(report["global_metrics"]["orientation_absolute_rad"]["maximum"]["value"], np.pi / 2)
    assert report["global_metrics"]["linear_velocity_absolute_m_s"]["maximum"]["value"] == 5
    assert report["global_metrics"]["angular_velocity_absolute_rad_s"]["maximum"]["value"] == 2
    assert all(x["available"] for x in report["cumulative_checkpoints"])
    assert report["cumulative_checkpoints"][0]["metrics"]["position_relative"]["maximum"]["value"] < 1
    tiny = {k: (x[:1].copy() if k == "time" else x[:1, :1, :1].copy()) for k, x in ref.items()}
    tiny["com_position"][:] = [1e-200, 0, 0]
    tiny_a = {k: x.copy() for k, x in tiny.items()}; tiny_a["com_position"] *= 2
    r, _ = analyze_trajectories(tiny_a, tiny, ["root"], "root")
    assert r["global_metrics"]["position_relative"]["maximum"]["value"] == 1
    assert r["global_metrics"]["height_relative"]["maximum"] is None
    bad = {k: x.copy() for k, x in native.items()}
    bad["com_position"][2, 1, 0] = np.nan; bad["orientation"][2, 1, 0] = 0
    r, _ = analyze_trajectories(bad, ref, ["root", "foot"], "root")
    assert r["global_metrics"]["position_relative"]["undefined_reasons"]["nonfinite_input"] == 1
    assert r["global_metrics"]["orientation_absolute_rad"]["undefined_reasons"]["zero_quaternion"] == 1
    json.dumps(r, allow_nan=False)
    bad["time"][1] += .001
    try:
        analyze_trajectories(bad, ref, ["root", "foot"], "root")
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched timestamps must be rejected")
    return dict(self_test="passed", cases=["analytic maximum/location/denominator", "zero and tiny denominators",
        "quaternion sign/short angle/zero norm", "absolute velocities", "cumulative maxima", "nonfinite input",
        "JSON without NaN", "exact timestamp alignment"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--newton", type=Path)
    parser.add_argument("--body-names", type=Path)
    parser.add_argument("--root-body")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(_self_test(), indent=2))
        return 0
    if any(x is None for x in (args.native, args.newton, args.body_names, args.output)):
        parser.error("--native --newton --body-names --output are required")
    meta = json.loads(args.body_names.read_text())
    names = meta if isinstance(meta, list) else meta["body_names"]
    metadata_root = None if isinstance(meta, list) else meta.get("root_body_name")
    if args.root_body and metadata_root and args.root_body != metadata_root:
        parser.error("--root-body conflicts with body_names.json root_body_name")
    root = args.root_body or metadata_root
    if root is None:
        parser.error("Provide --root-body or root_body_name in body_names.json")
    report, arrays = analyze_trajectories(_read_npz(args.native), _read_npz(args.newton), names, root)
    report["provenance"] = dict(
        files={key: dict(path=str(path.resolve()), sha256=_sha(path)) for key, path in
               (("native", args.native), ("newton", args.newton), ("body_names", args.body_names))},
        analyzer_sha256=_sha(Path(__file__)), host=platform.node(), python=sys.version, numpy=np.__version__)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    np.savez_compressed(args.output / "per-frame-errors.npz", **arrays)
    _write_readme(args.output / "README.md", report)
    print(json.dumps(dict(analysis_complete=True, frames=report["frames"], output=str(args.output)), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
