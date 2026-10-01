#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
batch_body_analysis.py

Scaled-up / automatic version of body_analysis.py. Point it at a folder of
DeepLabCut-style body-tracking CSVs (directly via CLI flags, or via a
config file) and it will process every matching file into a set of
kinematic measures: back-centroid displacement, body angle/angular
velocity, body length, hindlimb displacement, and tail angle/angular
velocity -- computed at the tracking data's native frame rate, then
block-averaged down to a target rate (e.g. to match your imaging data),
the same way pupil_area_norm is handled in batch_pupil_analysis.py.

As in body_analysis.py, bodyparts are auto-detected from each file
(list_bodyparts) rather than hard-coded -- only the six *roles* needed to
compute these composite measures (back_low, back_mid, hind_left,
hind_right, tail_base, tail_mid) are named, and those names are resolved
against whatever's actually in the file (see resolve_bodypart), so a
differently-labelled skeleton still works if you override the role names.

USAGE
-----
    python batch_body_analysis.py \\
        --input-dir /path/to/body_data/sub-waves04/video_preds \\
        --output-dir /path/to/out/body \\
        --fs 30 --target-fs 10

    python batch_body_analysis.py --config body_config.yaml

OUTPUT
------
For each input file <stem>.csv, writes <output_dir>/<stem>_body_measures.csv
with columns:
    frame, time_sec,
    back_center_x, back_center_y, back_center_fd,
    body_angle_deg, body_angular_velocity_degs,
    body_length_px, body_length_dt,
    hind_left_fd, hind_right_fd,
    tail_angle_deg, tail_angular_velocity_degs
A run_summary.csv logging success/failure and basic diagnostics is also
written.

NOTE ON DOWNSAMPLING: `fs` is the tracking data's NATIVE frame rate (e.g.
30 fps, same camera as the pupil video); `target_fs` is the rate every
measure is block-averaged down to (e.g. 10 Hz to match imaging). If your
data is already at a single fixed rate with nothing to downsample, set
fs == target_fs. `body_angle_deg`/`tail_angle_deg` are averaged with a
circular mean (correct across the +/-180 degree wraparound) rather than a
plain mean; every other column uses a plain mean of its native-fs values
within each block.

NOTE ON UNITS: the original walkthrough script computed body angular
velocity in radians/s but tail angular velocity in degrees/s. This script
reports BOTH in degrees/s for consistency -- flag this to yourself if
you're comparing directly against numbers from the old script.

NOTE ON DEFAULTS (if you used an earlier version of this script): `fs`
used to mean "the rate everything is already at" and was used directly for
angular velocity with no downsampling step (default 10, implicitly
assuming pre-downsampled 10 Hz data). It now means the tracking data's
NATIVE rate (default 30, matching the camera), and a separate `target_fs`
(default 10) controls the block-averaging. If your tracking really is
native 10 Hz already, pass --fs 10 --target-fs 10 to reproduce the old,
non-downsampling behaviour.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from keypoint_utils import (
    load_dlc_csv,
    list_bodyparts,
    extract_keypoint,
    resolve_bodypart,
    euclidean_distance,
    framewise_displacement,
    angle_deg,
    angular_velocity_deg,
    apply_likelihood_mask,
    trim_and_downsample,
)

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


LOGGER = logging.getLogger("batch_body_analysis")

DEFAULTS = dict(
    input_dir=None,
    pattern="*.csv",
    recursive=False,
    output_dir="body_output",
    # NOTE: as of this version, fs is the NATIVE camera sampling rate (Hz),
    # and target_fs is the rate everything gets downsampled to -- this
    # matches how the pupil script's fs/target_fs work, and how the two
    # were recorded (same camera). Previously this script (like the
    # original body_analysis.py walkthrough) used a single `fs` directly
    # for angular velocity with NO downsampling step, implicitly assuming
    # the data was already at 10 Hz -- if your tracking is actually native
    # 30 fps, that undercounted angular velocity by ~3x. If you really do
    # have data already at a single fixed rate with no downsampling
    # needed, set fs == target_fs.
    fs=30.0,                    # native camera sampling rate (Hz)
    target_fs=10.0,             # output rate (Hz) after block-averaging, e.g. to match imaging
    likelihood_threshold=None,  # optional: NaN-out x/y below this DLC likelihood
    back_low="back_low",
    back_mid="back_mid",
    hind_left="hind_left",
    hind_right="hind_right",
    tail_base="tail_base",
    tail_mid="tail_mid",
    save_plots=False,
)


# ---------------------------------------------------------------------
# Config handling
# ---------------------------------------------------------------------

def load_config_file(path) -> dict:
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        if not _HAS_YAML:
            raise RuntimeError(
                "PyYAML is not installed but a .yaml config was given. "
                "Install with `pip install pyyaml` or use a .json config."
            )
        return yaml.safe_load(text) or {}
    elif path.suffix.lower() == ".json":
        return json.loads(text)
    else:
        raise ValueError(f"Unsupported config extension: {path.suffix} (use .yaml/.yml/.json)")


def build_config(args: argparse.Namespace) -> dict:
    cfg = dict(DEFAULTS)
    if args.config:
        file_cfg = load_config_file(args.config)
        cfg.update({k: v for k, v in file_cfg.items() if v is not None})
    for key in DEFAULTS:
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            cfg[key] = cli_val
    if cfg["input_dir"] is None:
        raise ValueError("input_dir must be set via --input-dir or a config file")
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=str, default=None,
                   help="Path to a .yaml/.yml or .json config file")
    p.add_argument("--input-dir", dest="input_dir", type=str, default=None,
                   help="Directory containing body tracking CSVs")
    p.add_argument("--pattern", type=str, default=None,
                   help="Glob pattern for tracking CSVs (default '*.csv')")
    p.add_argument("--recursive", action="store_true", default=None,
                   help="Search input-dir recursively for files matching pattern")
    p.add_argument("--output-dir", dest="output_dir", type=str, default=None,
                   help="Where to write per-file measurement CSVs")
    p.add_argument("--fs", type=float, default=None,
                   help="Native camera sampling rate in Hz (default 30)")
    p.add_argument("--target-fs", dest="target_fs", type=float, default=None,
                   help="Output rate in Hz after block-averaging, e.g. to match imaging data (default 10)")
    p.add_argument("--likelihood-threshold", dest="likelihood_threshold", type=float,
                   default=None,
                   help="Optional: NaN-out x/y for frames below this DLC likelihood")
    p.add_argument("--back-low", dest="back_low", type=str, default=None,
                   help="Name/alias of the lower-back keypoint")
    p.add_argument("--back-mid", dest="back_mid", type=str, default=None,
                   help="Name/alias of the mid-back keypoint")
    p.add_argument("--hind-left", dest="hind_left", type=str, default=None,
                   help="Name/alias of the left hindlimb keypoint")
    p.add_argument("--hind-right", dest="hind_right", type=str, default=None,
                   help="Name/alias of the right hindlimb keypoint")
    p.add_argument("--tail-base", dest="tail_base", type=str, default=None,
                   help="Name/alias of the tail-base keypoint")
    p.add_argument("--tail-mid", dest="tail_mid", type=str, default=None,
                   help="Name/alias of the tail-mid keypoint")
    p.add_argument("--save-plots", dest="save_plots", action="store_true", default=None,
                   help="Save a diagnostic PNG (body angle + hindlimb displacement) per file")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return p.parse_args()


# ---------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------

def _load_role_xy(data, available, cfg, likelihood_threshold, role_key, role_label):
    name = resolve_bodypart(available, cfg[role_key], role_label)
    xy = extract_keypoint(data, name, coords=("x", "y"))
    if likelihood_threshold is not None:
        lh = extract_keypoint(data, name, coords=("likelihood",))
        xy = apply_likelihood_mask(xy, lh, likelihood_threshold)
    return xy


def process_body_file(csv_path: Path, cfg: dict):
    data = load_dlc_csv(csv_path)
    available = list_bodyparts(data)
    lt = cfg["likelihood_threshold"]

    back_low = _load_role_xy(data, available, cfg, lt, "back_low", "back_low")
    back_mid = _load_role_xy(data, available, cfg, lt, "back_mid", "back_mid")
    hind_left = _load_role_xy(data, available, cfg, lt, "hind_left", "hind_left")
    hind_right = _load_role_xy(data, available, cfg, lt, "hind_right", "hind_right")
    tail_base = _load_role_xy(data, available, cfg, lt, "tail_base", "tail_base")
    tail_mid = _load_role_xy(data, available, cfg, lt, "tail_mid", "tail_mid")

    n_frames_native = len(back_low)
    fs = cfg["fs"]
    target_fs = cfg["target_fs"]
    if fs / target_fs != round(fs / target_fs):
        LOGGER.warning(
            "fs/target_fs = %.3f is not an integer for %s; downsampling "
            "block size will be rounded -- check output alignment.",
            fs / target_fs, csv_path.name,
        )

    # --- Compute every measure at native fs (unchanged from the original
    # walkthrough's per-frame logic) ---

    # Back centroid + framewise displacement
    back_center = pd.DataFrame({
        "x": (back_mid["x"] + back_low["x"]) / 2,
        "y": (back_mid["y"] + back_low["y"]) / 2,
    })
    back_center_fd = framewise_displacement(back_center)

    # Body angle (back_mid -> back_low) + angular velocity. angular_velocity_deg
    # uses fs internally to convert the per-frame angle change into deg/s, so
    # it must be called with the NATIVE fs (the diff step happens between
    # native-fs frames) -- this is unchanged, only the meaning of `fs` in the
    # config is now explicit about being the native rate.
    body_angle = angle_deg(back_mid, back_low)
    body_angular_velocity = angular_velocity_deg(body_angle, fs)

    # Body length + rate of change
    body_length = euclidean_distance(back_low, back_mid)
    body_length_dt = body_length.diff()

    # Hindlimb framewise displacement
    hind_left_fd = framewise_displacement(hind_left)
    hind_right_fd = framewise_displacement(hind_right)

    # Tail angle (tail_base -> tail_mid) + angular velocity
    tail_angle = angle_deg(tail_base, tail_mid)
    tail_angular_velocity = angular_velocity_deg(tail_angle, fs)

    # --- Downsample every measure from native fs to target_fs by block-
    # averaging (same treatment as pupil_area_norm in the pupil script).
    # Angle columns use a circular mean since a plain mean is wrong across
    # the +/-180 degree wraparound; everything else uses a normal mean. ---
    def ds(series, circular=False):
        return trim_and_downsample(series, 0, fs, target_fs, circular=circular)

    back_center_x_ds = ds(back_center["x"])
    back_center_y_ds = ds(back_center["y"])
    back_center_fd_ds = ds(back_center_fd)
    body_angle_ds = ds(body_angle, circular=True)
    body_angular_velocity_ds = ds(body_angular_velocity)
    body_length_ds = ds(body_length)
    body_length_dt_ds = ds(body_length_dt)
    hind_left_fd_ds = ds(hind_left_fd)
    hind_right_fd_ds = ds(hind_right_fd)
    tail_angle_ds = ds(tail_angle, circular=True)
    tail_angular_velocity_ds = ds(tail_angular_velocity)

    n_out = len(back_center_x_ds)

    result = pd.DataFrame({
        "frame": np.arange(n_out),
        "time_sec": np.arange(n_out) / target_fs,
        "back_center_x": back_center_x_ds,
        "back_center_y": back_center_y_ds,
        "back_center_fd": back_center_fd_ds,
        "body_angle_deg": body_angle_ds,
        "body_angular_velocity_degs": body_angular_velocity_ds,
        "body_length_px": body_length_ds,
        "body_length_dt": body_length_dt_ds,
        "hind_left_fd": hind_left_fd_ds,
        "hind_right_fd": hind_right_fd_ds,
        "tail_angle_deg": tail_angle_ds,
        "tail_angular_velocity_degs": tail_angular_velocity_ds,
    })

    diagnostics = dict(
        n_frames_raw=int(n_frames_native),
        n_frames_out=int(n_out),
        mean_back_center_fd=float(np.nanmean(back_center_fd)),
        mean_body_length_px=float(np.nanmean(body_length)),
        pct_nan_back_low=float(back_low.isna().any(axis=1).mean() * 100),
        pct_nan_hind_left=float(hind_left.isna().any(axis=1).mean() * 100),
        pct_nan_hind_right=float(hind_right.isna().any(axis=1).mean() * 100),
    )
    return result, diagnostics


def save_diagnostic_plot(result: pd.DataFrame, out_path: Path, title: str):
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(result["time_sec"], result["body_angle_deg"], linewidth=1)
    axes[0].set_ylabel("Body angle (deg)")
    axes[1].plot(result["time_sec"], result["back_center_fd"], linewidth=1, color="tab:orange")
    axes[1].set_ylabel("Back centroid FD (px)")
    axes[2].plot(result["time_sec"], result["hind_left_fd"], linewidth=1, label="hind_left")
    axes[2].plot(result["time_sec"], result["hind_right_fd"], linewidth=1, label="hind_right")
    axes[2].set_ylabel("Hindlimb FD (px)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def find_input_files(cfg: dict) -> list:
    input_dir = Path(cfg["input_dir"])
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input_dir does not exist: {input_dir}")
    globber = input_dir.rglob if cfg["recursive"] else input_dir.glob
    return sorted(globber(cfg["pattern"]))


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    cfg = build_config(args)
    LOGGER.debug("Resolved config: %s", cfg)

    input_files = find_input_files(cfg)
    if not input_files:
        LOGGER.error("No files matched pattern '%s' in %s", cfg["pattern"], cfg["input_dir"])
        sys.exit(1)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    if cfg["save_plots"]:
        plot_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Found %d file(s) to process in %s", len(input_files), cfg["input_dir"])

    summary_rows = []
    n_ok = n_fail = 0

    for csv_path in input_files:
        LOGGER.info("Processing %s", csv_path.name)
        try:
            result, diag = process_body_file(csv_path, cfg)
            out_csv = output_dir / f"{csv_path.stem}_body_measures.csv"
            result.to_csv(out_csv, index=False)
            if cfg["save_plots"]:
                save_diagnostic_plot(
                    result, plot_dir / f"{csv_path.stem}_body_kinematics.png", csv_path.stem
                )
            summary_rows.append(dict(file=csv_path.name, status="ok",
                                      output=str(out_csv), **diag))
            n_ok += 1
        except Exception as exc:
            LOGGER.error("Failed on %s: %s", csv_path.name, exc)
            LOGGER.debug(traceback.format_exc())
            summary_rows.append(dict(file=csv_path.name, status="failed", error=str(exc)))
            n_fail += 1

    summary_path = output_dir / "run_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    LOGGER.info("Done: %d succeeded, %d failed. Summary written to %s",
                n_ok, n_fail, summary_path)

    if n_fail and not n_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
