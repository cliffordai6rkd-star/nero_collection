from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from nero_collection.cameras import CameraManager
from nero_collection.config import CollectionConfig, load_config
from nero_collection.h5_writer import EpisodeBuffer
from nero_collection.keyboard import TerminalKeys
from nero_collection.teleop.master_slave import MasterSlaveTeleop

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    if args.backend:
        config = _with_backend(config, args.backend)
    return run_collection(
        config=config,
        episode_limit=args.episode_limit,
        dry_run_duration_s=args.dry_run_duration,
        auto_save=args.auto_save or args.dry_run_duration is not None,
    )


def run_collection(
    config: CollectionConfig,
    episode_limit: int | None = None,
    dry_run_duration_s: float | None = None,
    auto_save: bool = False,
) -> int:
    teleop = MasterSlaveTeleop(config)
    cameras = CameraManager.from_config(config.cameras)
    output_dir = config.output.directory
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_index = _next_episode_index(output_dir, config.output.prefix)
    completed = 0

    try:
        teleop.start()
        cameras.start()
        with TerminalKeys() as keys:
            if not keys.is_tty and dry_run_duration_s is None:
                raise RuntimeError("stdin is not a TTY; use --dry-run-duration for non-interactive runs")

            while episode_limit is None or completed < episode_limit:
                _wait_for_record_start(teleop, keys, config, dry_run_duration_s)
                teleop.enter_teleop()
                buffer = EpisodeBuffer(
                    config=config,
                    arm_names=teleop.arm_names,
                    sample_rate_hz=config.teleop.command.sample_rate_hz,
                )
                log.info("recording episode %04d; press SPACE to stop", episode_index)
                print("Recording. Press SPACE to stop.", flush=True)
                _record_episode(buffer, teleop, cameras, keys, config, dry_run_duration_s)
                log.info("recorded %d teleop samples", buffer.sample_count)

                save = _wait_for_save_choice(keys, auto_save)
                if save:
                    path = _episode_path(output_dir, config.output.prefix, episode_index)
                    buffer.save(path)
                    log.info("saved episode to %s", path)
                    print(f"Saved: {path}", flush=True)
                    episode_index += 1
                    completed += 1
                else:
                    log.info("discarded episode %04d", episode_index)
                    print("Discarded this episode.", flush=True)
                    completed += 1
                if config.teleop.command.reset_after_episode:
                    print("Resetting and checking both arms...", flush=True)
                    teleop.reset_to_rest()
    except KeyboardInterrupt:
        print("\nCtrl-C received; shutting down.", flush=True)
        return 130
    finally:
        cameras.stop()
        teleop.shutdown()
    return 0


def _wait_for_record_start(
    teleop: MasterSlaveTeleop,
    keys: TerminalKeys,
    config: CollectionConfig,
    dry_run_duration_s: float | None,
) -> None:
    if dry_run_duration_s is not None:
        log.info("dry-run: auto-start recording")
        return
    print(
        "Press r to enter teleop and record, t to teleoperate without recording, or q to quit.",
        flush=True,
    )
    idle_period = 1.0 / max(config.teleop.command.idle_rate_hz, 1.0)
    while True:
        start = time.monotonic()
        key = keys.read_key(0.0)
        if key in {"r", "R"}:
            return
        if key in {"t", "T"}:
            teleop.enter_unrecorded_teleop()
            print("Unrecorded teleoperation active. Press r to start recording.", flush=True)
            continue
        if key in {"q", "Q", "\x03"}:
            raise KeyboardInterrupt
        teleop.idle_step()
        elapsed = time.monotonic() - start
        if elapsed < idle_period:
            time.sleep(idle_period - elapsed)


def _record_episode(
    buffer: EpisodeBuffer,
    teleop: MasterSlaveTeleop,
    cameras: CameraManager,
    keys: TerminalKeys,
    config: CollectionConfig,
    dry_run_duration_s: float | None,
) -> None:
    sample_period = 1.0 / max(config.teleop.command.sample_rate_hz, 1.0)
    start_t = time.monotonic()
    while True:
        loop_t = time.monotonic()
        key = keys.read_key(0.0)
        if key == " ":
            return
        if key in {"q", "Q", "\x03"}:
            raise KeyboardInterrupt
        if dry_run_duration_s is not None and loop_t - start_t >= dry_run_duration_s:
            return

        timestamp_us, values = teleop.teleop_step()
        buffer.append_teleop(timestamp_us, values)
        for frame in cameras.poll():
            buffer.append_camera(frame.camera_name, frame.timestamp_us, frame.frame)

        elapsed = time.monotonic() - loop_t
        if elapsed < sample_period:
            time.sleep(sample_period - elapsed)


def _wait_for_save_choice(keys: TerminalKeys, auto_save: bool) -> bool:
    if auto_save:
        log.info("auto-save enabled")
        return True
    print("Press y to save the data or n to discard it.", flush=True)
    while True:
        key = keys.read_key(0.1)
        if key in {"y", "Y"}:
            return True
        if key in {"n", "N"}:
            return False
        if key in {"q", "Q", "\x03"}:
            raise KeyboardInterrupt


def _episode_path(output_dir: Path, prefix: str, index: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{prefix}_{index:04d}_{stamp}.h5"


def _next_episode_index(output_dir: Path, prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)_.*\.h5$")
    max_index = -1
    for path in output_dir.glob(f"{prefix}_*.h5"):
        match = pattern.match(path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def _with_backend(config: CollectionConfig, backend: str) -> CollectionConfig:
    return replace(config, teleop=replace(config.teleop, backend=backend))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nero master-slave teleoperation data collection.")
    parser.add_argument(
        "--config",
        default="configs/master_slave_can.yaml",
        help="Path to collection YAML config.",
    )
    parser.add_argument(
        "--backend",
        choices=("pyagxarm", "mock"),
        help="Override teleop.backend from the config.",
    )
    parser.add_argument(
        "--episode-limit",
        type=int,
        help="Stop after this many accepted or discarded episodes.",
    )
    parser.add_argument(
        "--dry-run-duration",
        type=float,
        help="Non-interactive run duration in seconds; automatically starts and saves.",
    )
    parser.add_argument(
        "--auto-save",
        action="store_true",
        help="Save each stopped episode without asking y/n.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
