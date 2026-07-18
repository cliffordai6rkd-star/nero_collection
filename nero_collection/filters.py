from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from nero_collection.config import StateParamConfig


@dataclass
class OnePoleLowPass:
    cutoff_hz: float
    median_window: int = 1
    state: np.ndarray | None = None
    previous_timestamp_us: int | None = None
    history: deque[np.ndarray] = field(default_factory=deque)

    def __post_init__(self) -> None:
        if not np.isfinite(self.cutoff_hz) or self.cutoff_hz <= 0:
            raise ValueError("cutoff_hz must be positive and finite")
        if self.median_window < 1 or self.median_window % 2 == 0:
            raise ValueError("median_window must be a positive odd integer")

    def apply(self, value: np.ndarray, timestamp_us: int) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        timestamp_us = int(timestamp_us)
        if self.state is not None and value.shape != self.state.shape:
            raise ValueError(f"filter shape changed from {self.state.shape} to {value.shape}")
        self.history.append(value.copy())
        while len(self.history) > self.median_window:
            self.history.popleft()
        samples = list(self.history)
        if samples:
            samples = [samples[0]] * (self.median_window - len(samples)) + samples
        median_value = np.median(np.stack(samples, axis=0), axis=0)
        if self.state is None:
            self.state = median_value.copy()
            self.previous_timestamp_us = timestamp_us
            return self.state.copy()
        assert self.previous_timestamp_us is not None
        dt = (timestamp_us - self.previous_timestamp_us) * 1e-6
        if dt <= 0:
            raise ValueError(
                f"filter timestamps must be strictly increasing: "
                f"{timestamp_us} <= {self.previous_timestamp_us}"
            )
        alpha = 1.0 - np.exp(-2.0 * np.pi * self.cutoff_hz * dt)
        self.state = alpha * median_value + (1.0 - alpha) * self.state
        self.previous_timestamp_us = timestamp_us
        return self.state.copy()


@dataclass
class LowPassVelocityDifferentiator:
    """Estimate acceleration from velocity samples on the recorded timeline."""

    cutoff_hz: float | None
    filtered_velocity: np.ndarray | None = None
    previous_timestamp_us: int | None = None

    def __post_init__(self) -> None:
        if self.cutoff_hz is not None and self.cutoff_hz <= 0:
            raise ValueError("cutoff_hz must be positive when provided")

    def apply(self, velocity: np.ndarray, timestamp_us: int) -> np.ndarray:
        velocity = np.asarray(velocity, dtype=np.float64)
        timestamp_us = int(timestamp_us)

        if self.filtered_velocity is None:
            self.filtered_velocity = velocity.copy()
            self.previous_timestamp_us = timestamp_us
            return np.zeros_like(velocity)
        if velocity.shape != self.filtered_velocity.shape:
            raise ValueError(
                f"velocity shape changed from {self.filtered_velocity.shape} to {velocity.shape}"
            )

        assert self.previous_timestamp_us is not None
        dt = (timestamp_us - self.previous_timestamp_us) * 1e-6
        if dt <= 0:
            raise ValueError(
                f"velocity timestamps must be strictly increasing: "
                f"{timestamp_us} <= {self.previous_timestamp_us}"
            )

        previous_velocity = self.filtered_velocity
        if self.cutoff_hz is None:
            filtered_velocity = velocity.copy()
        else:
            alpha = 1.0 - np.exp(-2.0 * np.pi * self.cutoff_hz * dt)
            filtered_velocity = previous_velocity + alpha * (velocity - previous_velocity)

        acceleration = (filtered_velocity - previous_velocity) / dt
        self.filtered_velocity = filtered_velocity
        self.previous_timestamp_us = timestamp_us
        return acceleration


@dataclass
class DatasetFilterBank:
    state_params: dict[str, StateParamConfig]
    filters: dict[str, OnePoleLowPass] = field(default_factory=dict)

    def apply(
        self,
        dataset_name: str,
        state_name: str,
        value: np.ndarray,
        timestamp_us: int,
    ) -> np.ndarray:
        param = self.state_params.get(state_name)
        if not param or not param.lowpass:
            return value
        filt = self.filters.get(dataset_name)
        if filt is None:
            filt = OnePoleLowPass(param.lowpass_cutoff_hz, param.median_window)
            self.filters[dataset_name] = filt
        return filt.apply(value, timestamp_us)
