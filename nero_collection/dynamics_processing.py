from __future__ import annotations

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import butter, medfilt, sosfiltfilt


def reconstruct_state_from_positions(
    timestamp_us: np.ndarray,
    q: np.ndarray,
    *,
    state_method: str,
    spline_smoothing_rad2: float,
    fourier_fundamental_hz: float,
    fourier_harmonics: int,
    evaluation_timestamp_us: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_t, source_q = _unique_samples(timestamp_us, q, minimum=4)
    evaluation_t = (
        source_t
        if evaluation_timestamp_us is None
        else _timestamp_vector(evaluation_timestamp_us, "evaluation_timestamp_us")
    )
    origin_us = source_t[0]
    source_time_s = (source_t - origin_us).astype(np.float64) * 1e-6
    evaluation_t = np.clip(evaluation_t, source_t[0], source_t[-1])
    evaluation_time_s = (evaluation_t - origin_us).astype(np.float64) * 1e-6
    if state_method == "spline":
        return _spline_state(
            source_time_s,
            source_q,
            evaluation_time_s,
            spline_smoothing_rad2,
        )
    if state_method == "fourier":
        return _fourier_state(
            source_time_s,
            source_q,
            evaluation_time_s,
            fourier_fundamental_hz,
            fourier_harmonics,
        )
    raise ValueError(f"unsupported state reconstruction method: {state_method!r}")


def resample_columns(
    source_timestamp_us: np.ndarray,
    values: np.ndarray,
    target_timestamp_us: np.ndarray,
    *,
    fallback_source_timestamp_us: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("resampling values must be a finite (N, D) array")
    target = _timestamp_vector(target_timestamp_us, "target_timestamp_us")
    source = np.asarray(source_timestamp_us, dtype=np.int64)
    if source.ndim == 1:
        source = np.repeat(source[:, None], values.shape[1], axis=1)
    if source.shape != values.shape:
        raise ValueError(
            f"source timestamps must have shape {(values.shape[0],)} or {values.shape}; got {source.shape}"
        )
    result = np.empty((target.size, values.shape[1]), dtype=np.float64)
    target_float = target.astype(np.float64)
    fallback = None
    if fallback_source_timestamp_us is not None:
        fallback = np.asarray(fallback_source_timestamp_us, dtype=np.int64)
        if fallback.ndim == 1:
            fallback = np.repeat(fallback[:, None], values.shape[1], axis=1)
        if fallback.shape != values.shape:
            raise ValueError(
                f"fallback source timestamps must have shape {values.shape}; got {fallback.shape}"
            )
    for column in range(values.shape[1]):
        try:
            column_t, column_values = _unique_samples(
                source[:, column],
                values[:, column],
                minimum=2,
            )
        except ValueError:
            if fallback is None:
                raise
            column_t, column_values = _unique_samples(
                fallback[:, column],
                values[:, column],
                minimum=2,
            )
        result[:, column] = np.interp(
            target_float,
            column_t.astype(np.float64),
            column_values,
        )
    return result


def select_source_timestamps(
    primary_timestamp_us: np.ndarray,
    acquired_timestamp_us: np.ndarray,
    *,
    minimum_unique: int,
) -> tuple[np.ndarray, bool]:
    primary = np.asarray(primary_timestamp_us, dtype=np.int64).reshape(-1)
    acquired = _timestamp_vector(acquired_timestamp_us, "acquired_timestamp_us")
    if primary.size != acquired.size:
        raise ValueError("primary and acquired timestamp series must have the same length")
    primary_valid = (
        np.all(primary > 0)
        and np.all(np.diff(primary) >= 0)
        and np.unique(primary).size >= minimum_unique
    )
    return (primary.copy(), False) if primary_valid else (acquired.copy(), True)


def filter_torque(
    timestamp_us: np.ndarray,
    tau: np.ndarray,
    *,
    median_window: int,
    lowpass_hz: float,
) -> np.ndarray:
    timestamp_us = _timestamp_vector(timestamp_us, "timestamp_us")
    tau = np.asarray(tau, dtype=np.float64)
    if tau.ndim != 2 or tau.shape[0] != timestamp_us.size or not np.isfinite(tau).all():
        raise ValueError("torque filtering requires finite tau with shape (N, D)")
    if median_window < 1 or median_window % 2 == 0:
        raise ValueError("median_window must be a positive odd integer")
    filtered = np.column_stack(
        [medfilt(tau[:, joint], kernel_size=median_window) for joint in range(tau.shape[1])]
    )
    dt = np.diff(timestamp_us).astype(np.float64) * 1e-6
    if np.any(dt <= 0):
        raise ValueError("torque filtering requires strictly increasing timestamps")
    sample_rate = 1.0 / float(np.median(dt))
    nyquist = 0.5 * sample_rate
    if lowpass_hz >= nyquist * 0.95:
        return filtered
    sos = butter(4, lowpass_hz / nyquist, btype="low", output="sos")
    return sosfiltfilt(sos, filtered, axis=0)


def _spline_state(time_s, q, evaluation_time_s, smoothing_rad2):
    shape = (evaluation_time_s.size, q.shape[1])
    q_fit = np.empty(shape, dtype=np.float64)
    dq = np.empty(shape, dtype=np.float64)
    ddq = np.empty(shape, dtype=np.float64)
    smoothing = float(smoothing_rad2) * time_s.size
    for joint in range(q.shape[1]):
        spline = UnivariateSpline(time_s, q[:, joint], k=3, s=smoothing)
        q_fit[:, joint] = spline(evaluation_time_s)
        dq[:, joint] = spline.derivative(1)(evaluation_time_s)
        ddq[:, joint] = spline.derivative(2)(evaluation_time_s)
    return q_fit, dq, ddq


def _fourier_state(time_s, q, evaluation_time_s, fundamental_hz, harmonics):
    omega = 2.0 * np.pi * float(fundamental_hz) * np.arange(1, harmonics + 1)
    source_phase = time_s[:, None] * omega[None, :]
    source_sin = np.sin(source_phase)
    source_cos = np.cos(source_phase)
    design = np.column_stack((np.ones(time_s.size), source_sin, source_cos))
    coefficients, _, _, _ = np.linalg.lstsq(design, q, rcond=None)
    sin_coefficients = coefficients[1 : harmonics + 1]
    cos_coefficients = coefficients[harmonics + 1 :]

    evaluation_phase = evaluation_time_s[:, None] * omega[None, :]
    sin_phase = np.sin(evaluation_phase)
    cos_phase = np.cos(evaluation_phase)
    evaluation_design = np.column_stack(
        (np.ones(evaluation_time_s.size), sin_phase, cos_phase)
    )
    q_fit = evaluation_design @ coefficients
    dq = cos_phase @ (sin_coefficients * omega[:, None]) - sin_phase @ (
        cos_coefficients * omega[:, None]
    )
    ddq = -sin_phase @ (sin_coefficients * omega[:, None] ** 2) - cos_phase @ (
        cos_coefficients * omega[:, None] ** 2
    )
    return q_fit, dq, ddq


def _unique_samples(timestamp_us, values, minimum):
    timestamp = _timestamp_vector(timestamp_us, "timestamp_us", strictly_increasing=False)
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] != timestamp.size or not np.isfinite(values).all():
        raise ValueError("sample values must be finite and match the timestamp length")
    if np.any(np.diff(timestamp) < 0):
        raise ValueError("sample timestamps must be non-decreasing")
    unique, first, counts = np.unique(timestamp, return_index=True, return_counts=True)
    last = first + counts - 1
    if unique.size < minimum:
        raise ValueError(f"at least {minimum} unique timestamped samples are required")
    return unique, values[last]


def _timestamp_vector(value, name, *, strictly_increasing=True):
    timestamp = np.asarray(value, dtype=np.int64).reshape(-1)
    if timestamp.size == 0 or np.any(timestamp <= 0):
        raise ValueError(f"{name} must contain positive timestamps")
    differences = np.diff(timestamp)
    if strictly_increasing and np.any(differences <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return timestamp
