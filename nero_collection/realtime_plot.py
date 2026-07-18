from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time
from collections import deque
from importlib.util import find_spec

import numpy as np

from nero_collection.config import RealtimePlotConfig, StateParamConfig
from nero_collection.contact_wrench import PinocchioJointTorqueResidualEstimator

log = logging.getLogger(__name__)


class SlidingJointBuffer:
    def __init__(self, window_s: float) -> None:
        self.window_s = float(window_s)
        self._timestamps_s: deque[float] = deque()
        self._q: deque[np.ndarray] = deque()
        self._tau: deque[np.ndarray] = deque()
        self._tau_ext: deque[np.ndarray] = deque()

    def append(
        self,
        timestamp_us: int,
        q: np.ndarray,
        tau: np.ndarray,
        tau_ext: np.ndarray,
    ) -> None:
        q = _plot_vector("q", q, 7)
        tau = _plot_vector("tau", tau, 7)
        tau_ext = _plot_vector("tau_ext", tau_ext, 7)
        timestamp_s = int(timestamp_us) / 1_000_000.0
        if self._timestamps_s and timestamp_s <= self._timestamps_s[-1]:
            return
        self._timestamps_s.append(timestamp_s)
        self._q.append(q)
        self._tau.append(tau)
        self._tau_ext.append(tau_ext)

        cutoff_s = timestamp_s - self.window_s
        while self._timestamps_s and self._timestamps_s[0] < cutoff_s:
            self._timestamps_s.popleft()
            self._q.popleft()
            self._tau.popleft()
            self._tau_ext.popleft()

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self._timestamps_s:
            return (
                np.empty((0,), dtype=np.float64),
                np.empty((0, 7), dtype=np.float64),
                np.empty((0, 7), dtype=np.float64),
                np.empty((0, 7), dtype=np.float64),
            )
        timestamps = np.asarray(self._timestamps_s, dtype=np.float64)
        return (
            timestamps - timestamps[-1],
            np.stack(self._q, axis=0),
            np.stack(self._tau, axis=0),
            np.stack(self._tau_ext, axis=0),
        )


class RealtimeJointPlotter:
    _REQUIRED_DATASETS = (
        "q_follower",
        "dq_follower",
        "ddq_follower",
        "tau_follower",
    )

    def __init__(
        self,
        config: RealtimePlotConfig,
        robot_states: dict[str, StateParamConfig],
    ) -> None:
        self.config = config
        self.robot_states = robot_states
        self._queue = None
        self._process = None
        self._closed = False
        self._process_failure_logged = False

    def start(self) -> None:
        if not self.config.enabled:
            return
        self._validate_required_states()
        if find_spec("matplotlib") is None:
            raise RuntimeError(
                "realtime_plot.enabled=true requires matplotlib; install matplotlib>=3.7"
            )
        if find_spec("pinocchio") is None:
            raise RuntimeError(
                "realtime_plot.enabled=true requires Pinocchio; install pin>=3,<4"
            )
        if not self.config.inverse_dynamics.urdf_path.is_file():
            raise RuntimeError(
                f"Inverse-dynamics URDF does not exist: {self.config.inverse_dynamics.urdf_path}"
            )
        context = mp.get_context("spawn")
        self._queue = context.Queue(maxsize=512)
        self._process = context.Process(
            target=_plot_worker,
            args=(self.config, self._queue),
            name="nero-realtime-plot",
            daemon=True,
        )
        self._process.start()
        log.info(
            "realtime plot process started datasets=q_follower,tau_follower,tau_ext "
            "window=%.1fs delay=%.3fs update=%.1fHz",
            self.config.window_s,
            self.config.inverse_dynamics.delay_s,
            self.config.update_rate_hz,
        )

    def append(self, timestamp_us: int, values: dict[str, tuple[str, np.ndarray]]) -> None:
        if not self.config.enabled or self._closed:
            return
        if self._process is None or self._queue is None:
            raise RuntimeError("Realtime plot has not been started")
        if not self._process.is_alive():
            self._closed = True
            if not self._process_failure_logged:
                log.info("realtime plot window closed or unavailable; collection continues")
                self._process_failure_logged = True
            return

        missing = [name for name in self._REQUIRED_DATASETS if name not in values]
        if missing:
            raise RuntimeError(f"Realtime tau_ext plot is missing teleop datasets: {missing}")
        sample = (
            int(timestamp_us),
            _plot_vector("q", values["q_follower"][1], 7),
            _plot_vector("dq", values["dq_follower"][1], 7),
            _plot_vector("ddq", values["ddq_follower"][1], 7),
            _plot_vector("tau", values["tau_follower"][1], 7),
        )
        try:
            self._queue.put_nowait(sample)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(sample)
            except queue.Full:
                pass

    def close(self) -> None:
        if self._queue is not None:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
        if self._process is not None:
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
        if self._queue is not None:
            self._queue.close()
        self._queue = None
        self._process = None
        self._closed = True

    def _validate_required_states(self) -> None:
        required = ("q", "velocity", "acceleration", "torque")
        disabled = [
            name
            for name in required
            if name not in self.robot_states or not self.robot_states[name].enabled
        ]
        if disabled:
            raise RuntimeError(
                "realtime tau_ext plot requires enabled robot_states "
                f"q, velocity, acceleration, torque; missing={disabled}"
            )


class _MatplotlibPlotWindow:
    _PLOTS = (
        ("q", "q [rad]", tuple(f"J{index}" for index in range(1, 8))),
        ("tau", "tau [N.m]", tuple(f"J{index}" for index in range(1, 8))),
        ("tau_ext", "tau_id - tau [N.m]", tuple(f"J{index}" for index in range(1, 8))),
    )

    def __init__(self, config: RealtimePlotConfig) -> None:
        import matplotlib.pyplot as plt

        self.config = config
        self.buffer = SlidingJointBuffer(config.window_s)
        self.plt = plt
        plt.ion()
        self.figure, axes = plt.subplots(1, 3, figsize=(17, 5), sharex=True)
        try:
            self.figure.canvas.manager.set_window_title("Nero realtime inverse-dynamics residual")
        except AttributeError:
            pass
        colors = plt.get_cmap("tab10").colors
        line_groups: list[tuple[object, ...]] = []
        for axis, (title, ylabel, labels) in zip(axes, self._PLOTS):
            lines = tuple(
                axis.plot([], [], color=colors[index], linewidth=1.1, label=label)[0]
                for index, label in enumerate(labels)
            )
            axis.set_title(title)
            axis.set_xlabel(f"time [s], delayed {config.inverse_dynamics.delay_s:.1f}s")
            axis.set_ylabel(ylabel)
            axis.set_xlim(-config.window_s, 0.0)
            axis.grid(True, alpha=0.25)
            axis.legend(loc="upper left", ncol=2, fontsize=8)
            line_groups.append(lines)
        self.axes = tuple(axes)
        self.lines = tuple(line_groups)
        self.figure.tight_layout()
        self.figure.show()
        self.process_events()

    def append(self, sample: tuple[int, np.ndarray, np.ndarray, np.ndarray]) -> None:
        self.buffer.append(*sample)

    def render(self) -> None:
        relative_time, q, tau, tau_ext = self.buffer.arrays()
        if relative_time.size:
            for axis, lines, data in zip(self.axes, self.lines, (q, tau, tau_ext)):
                for index, line in enumerate(lines):
                    line.set_data(relative_time, data[:, index])
                axis.set_xlim(-self.config.window_s, 0.0)
                _set_dynamic_ylim(axis, data)
            self.figure.canvas.draw()
        self.process_events()

    def process_events(self) -> None:
        self.figure.canvas.flush_events()

    def is_open(self) -> bool:
        return bool(self.plt.fignum_exists(self.figure.number))

    def close(self) -> None:
        self.plt.close(self.figure)


def _plot_worker(config: RealtimePlotConfig, sample_queue) -> None:
    try:
        estimator = PinocchioJointTorqueResidualEstimator(config.inverse_dynamics)
        window = _MatplotlibPlotWindow(config)
    except Exception:
        log.exception("failed to start realtime tau_ext plot")
        return
    pending: deque[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = deque()
    next_render_t = time.monotonic()
    next_diagnostic_t = time.monotonic()
    stop = False
    try:
        while not stop and window.is_open():
            timeout_s = max(0.0, min(0.05, next_render_t - time.monotonic()))
            try:
                item = sample_queue.get(timeout=timeout_s)
                received_item = True
            except queue.Empty:
                item = None
                received_item = False
            if received_item and item is None:
                stop = True
            elif received_item:
                pending.append(item)
                while True:
                    try:
                        queued_item = sample_queue.get_nowait()
                    except queue.Empty:
                        break
                    if queued_item is None:
                        stop = True
                        break
                    pending.append(queued_item)

            if pending:
                cutoff_us = pending[-1][0] - int(round(config.inverse_dynamics.delay_s * 1_000_000))
                while pending and pending[0][0] <= cutoff_us:
                    timestamp_us, q, dq, ddq, tau = pending.popleft()
                    estimate = estimator.estimate(q, dq, ddq, tau)
                    window.append((timestamp_us, q, tau, estimate.tau_residual))
                    now = time.monotonic()
                    if now >= next_diagnostic_t:
                        log.debug(
                            "tau_ext max_abs=%.4fNm",
                            float(np.max(np.abs(estimate.tau_residual))),
                        )
                        next_diagnostic_t = now + 2.0

            now = time.monotonic()
            if now >= next_render_t:
                window.render()
                next_render_t = now + 1.0 / config.update_rate_hz
            else:
                window.process_events()
    except Exception:
        log.exception("realtime tau_ext estimation failed")
    finally:
        window.close()


def _plot_vector(name: str, value: np.ndarray, size: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size != size or not np.isfinite(vector).all():
        raise RuntimeError(f"Realtime plot requires a finite {size}D {name} vector; got {vector}")
    return vector.copy()


def _set_dynamic_ylim(axis, data: np.ndarray) -> None:
    data_min = float(np.min(data))
    data_max = float(np.max(data))
    span = data_max - data_min
    padding = max(span * 0.08, 1e-3)
    if span < 1e-9:
        padding = max(abs(data_min) * 0.08, 0.05)
    axis.set_ylim(data_min - padding, data_max + padding)
