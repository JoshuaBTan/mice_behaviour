#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
batch_pupil_analysis.py

Scaled-up / automatic version of pupil_analysis.py. Instead of hand-editing
paths and re-running the script per video, point it at a folder of
DeepLabCut-style pupil-tracking CSVs (directly via CLI flags, or via a
config file) and it will process every matching file into a normalised
pupil-area time series, downsampled to match your imaging data.

Keypoints are resolved by name/alias against whatever bodyparts are
actually present in each file (see resolve_bodypart in keypoint_utils.py),
so you don't need to hard-code exact DLC label strings -- just the
defaults below, or override them if your labelling scheme differs.

USAGE
-----
    # plain CLI, single default imaging-start frame for every file
    python batch_pupil_analysis.py \\
        --input-dir /path/to/pupil_data/sub-waves04/video_preds \\
        --output-dir /path/to/out/pupil \\
        --fs 30 --target-fs 10 --imaging-start 66

    # per-video imaging-start frames (LED flash timing varies per video),
    # via a JSON file mapping {file_stem_or_name: start_frame}
    python batch_pupil_analysis.py \\
        --input-dir /path/to/pupil_data/sub-waves04/video_preds \\
        --output-dir /path/to/out/pupil \\
        --fs 30 --target-fs 10 --imaging-start-file imaging_start_times.json

    # via config file (CLI flags override config values)
    python batch_pupil_analysis.py --config pupil_config.yaml

    # override just one thing from the config
    python batch_pupil_analysis.py --config pupil_config.yaml --target-fs 20

IMAGING START
-------------
Imaging start (the LED-flash frame) is resolved per file, in order:
  1. an entry for that file's stem (or full filename) in --imaging-start-file
  2. --imaging-start as a fallback default
A file with no imaging_start_file entry and no default set fails (logged
in run_summary.csv) rather than silently using the wrong frame.

OUTPUT
------
For each input file <stem>.csv, writes <output_dir>/<stem>_pupil_measures.csv
with columns: frame, time_sec, pupil_area_norm (plus vertical_aperture_px /
nose_displacement_px if those keypoints are present). A run_summary.csv
logging success/failure, the imaging_start actually used, and basic
diagnostics per file is also written.
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
matplotlib.use("Agg")  # batch mode: never try to pop up a GUI window
import matplotlib.pyplot as plt

from keypoint_utils import (
    load_dlc_csv,
    list_bodyparts,
    extract_keypoint,
    resolve_bodypart,
    try_resolve_bodypart,
    euclidean_distance,
    framewise_displacement,
    apply_likelihood_mask,
    trim_and_downsample,
)

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


LOGGER = logging.getLogger("batch_pupil_analysis")

DEFAULTS = dict(
    input_dir=None,
    pattern="*.csv",
    recursive=False,
    output_dir="pupil_output",
    fs=30.0,                     # native camera sampling rate (Hz)
    target_fs=10.0,              # downsampled output rate (Hz), e.g. to match imaging
    # Frame index where imaging/behaviour acquisition starts (LED flash frame).
    # Two ways to set this, which can be combined:
    #   imaging_start_file : JSON mapping {file_stem_or_name: start_frame},
    #                        for when the LED timing differs per video
    #   imaging_start      : a single default value, used as a fallback for
    #                        any file not found in imaging_start_file (or as
    #                        the only source if you don't need per-file values)
    imaging_start=None,
    imaging_start_file=None,
    baseline_start_sec=5.0,      # baseline window, seconds after imaging_start
    baseline_end_sec=10.0,
    diameter_jump_threshold=10.0,  # px; frame-to-frame jumps above this -> NaN
    likelihood_threshold=None,   # optional: NaN-out x/y below this DLC likelihood
    left="leftpupil",
    right="rightpupil",
    dorsal="dorsalpupil",
    ventral="ventralpupil",
    # Optional supplementary keypoints/measures. These are resolved
    # leniently: if the keypoint isn't found in a given file, that one
    # measure is skipped (logged as a warning) rather than failing the
    # whole file, since not every dataset will have eyelid/nose tracking.
    compute_vertical_aperture=True,
    dorsal_eyelid="dorsaleyelid",
    ventral_eyelid="ventraleyelid",
    compute_nose_motion=True,
    nose="nose",
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
    # CLI flags win, but only if the user actually passed them
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
                   help="Directory containing pupil tracking CSVs")
    p.add_argument("--pattern", type=str, default=None,
                   help="Glob pattern for tracking CSVs (default '*.csv')")
    p.add_argument("--recursive", action="store_true", default=None,
                   help="Search input-dir recursively for files matching pattern")
    p.add_argument("--output-dir", dest="output_dir", type=str, default=None,
                   help="Where to write per-file measurement CSVs")
    p.add_argument("--fs", type=float, default=None,
                   help="Native camera sampling rate in Hz (default 30)")
    p.add_argument("--target-fs", dest="target_fs", type=float, default=None,
                   help="Downsampled output rate in Hz to match imaging data (default 10)")
    p.add_argument("--imaging-start", dest="imaging_start", type=int, default=None,
                   help="Default frame index at which imaging/behaviour acquisition starts, "
                        "used as a fallback for files not found in --imaging-start-file")
    p.add_argument("--imaging-start-file", dest="imaging_start_file", type=str, default=None,
                   help="Path to a JSON file mapping {file_stem_or_name: start_frame} for "
                        "per-video imaging-start frames (e.g. from manually checking the LED "
                        "flash). Falls back to --imaging-start for any file not listed.")
    p.add_argument("--baseline-start-sec", dest="baseline_start_sec", type=float, default=None,
                   help="Baseline window start, seconds after imaging_start")
    p.add_argument("--baseline-end-sec", dest="baseline_end_sec", type=float, default=None,
                   help="Baseline window end, seconds after imaging_start")
    p.add_argument("--diameter-jump-threshold", dest="diameter_jump_threshold",
                   type=float, default=None,
                   help="Frame-to-frame diameter change (px) above which a sample "
                        "is flagged as a tracking artifact and set to NaN")
    p.add_argument("--likelihood-threshold", dest="likelihood_threshold", type=float,
                   default=None,
                   help="Optional: NaN-out x/y for frames below this DLC likelihood")
    p.add_argument("--left", type=str, default=None, help="Name/alias of the left pupil keypoint")
    p.add_argument("--right", type=str, default=None, help="Name/alias of the right pupil keypoint")
    p.add_argument("--dorsal", type=str, default=None, help="Name/alias of the dorsal pupil keypoint")
    p.add_argument("--ventral", type=str, default=None, help="Name/alias of the ventral pupil keypoint")
    p.add_argument("--dorsal-eyelid", dest="dorsal_eyelid", type=str, default=None,
                   help="Name/alias of the dorsal eyelid keypoint (default 'dorsaleyelid')")
    p.add_argument("--ventral-eyelid", dest="ventral_eyelid", type=str, default=None,
                   help="Name/alias of the ventral eyelid keypoint (default 'ventraleyelid')")
    p.add_argument("--skip-vertical-aperture", dest="compute_vertical_aperture",
                   action="store_false", default=None,
                   help="Don't compute vertical eyelid aperture even if the keypoints exist")
    p.add_argument("--nose", type=str, default=None,
                   help="Name/alias of the nose keypoint (default 'nose')")
    p.add_argument("--skip-nose-motion", dest="compute_nose_motion",
                   action="store_false", default=None,
                   help="Don't compute nose framewise displacement even if the keypoint exists")
    p.add_argument("--save-plots", dest="save_plots", action="store_true", default=None,
                   help="Save a diagnostic PNG of the normalised pupil-area trace per file")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return p.parse_args()


# ---------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------

def load_imaging_start_map(path) -> dict:
    """
    Load a JSON file mapping {file_stem_or_name: start_frame}, e.g.:

        {
          "sub-waves04_ses-3_task-rest_run-1_face": 66,
          "sub-waves04_ses-3_task-rest_run-2_face": 82
        }

    Keys can be the bare file stem (no extension, recommended -- matches
    what shows up in run_summary.csv/output filenames) or the full
    filename including extension; both are checked at lookup time.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"imaging_start_file must contain a JSON object mapping "
            f"filename/stem -> start frame, got {type(data).__name__}"
        )
    return data


def resolve_imaging_start(csv_path: Path, cfg: dict, imaging_start_map: dict) -> int:
    """
    Resolve the imaging-start frame for one file, in order of preference:
      1. exact match on the file stem in imaging_start_map
      2. exact match on the full filename in imaging_start_map
      3. cfg['imaging_start'] as a fallback default, if set
    Raises ValueError if none of these resolve, so a missing entry fails
    loudly for that one file rather than silently using the wrong frame.
    """
    if imaging_start_map:
        if csv_path.stem in imaging_start_map:
            return int(imaging_start_map[csv_path.stem])
        if csv_path.name in imaging_start_map:
            return int(imaging_start_map[csv_path.name])

    if cfg["imaging_start"] is not None:
        if imaging_start_map:
            LOGGER.warning(
                "No imaging_start entry for %s in %s -- falling back to "
                "default imaging_start=%s",
                csv_path.name, cfg["imaging_start_file"], cfg["imaging_start"],
            )
        return int(cfg["imaging_start"])

    raise ValueError(
        f"No imaging_start found for {csv_path.name}: not present in "
        f"imaging_start_file ({cfg['imaging_start_file']!r}) and no default "
        f"imaging_start set in config/CLI."
    )


def process_pupil_file(csv_path: Path, cfg: dict, imaging_start: int):
    """Process a single tracking CSV into a normalised pupil-area trace."""
    data = load_dlc_csv(csv_path)
    available = list_bodyparts(data)

    left_name = resolve_bodypart(available, cfg["left"], "left")
    right_name = resolve_bodypart(available, cfg["right"], "right")
    dorsal_name = resolve_bodypart(available, cfg["dorsal"], "dorsal")
    ventral_name = resolve_bodypart(available, cfg["ventral"], "ventral")

    left = extract_keypoint(data, left_name, coords=("x", "y"))
    right = extract_keypoint(data, right_name, coords=("x", "y"))
    dorsal = extract_keypoint(data, dorsal_name, coords=("x", "y"))
    ventral = extract_keypoint(data, ventral_name, coords=("x", "y"))

    if cfg["likelihood_threshold"] is not None:
        left = apply_likelihood_mask(
            left, extract_keypoint(data, left_name, coords=("likelihood",)),
            cfg["likelihood_threshold"])
        right = apply_likelihood_mask(
            right, extract_keypoint(data, right_name, coords=("likelihood",)),
            cfg["likelihood_threshold"])
        dorsal = apply_likelihood_mask(
            dorsal, extract_keypoint(data, dorsal_name, coords=("likelihood",)),
            cfg["likelihood_threshold"])
        ventral = apply_likelihood_mask(
            ventral, extract_keypoint(data, ventral_name, coords=("likelihood",)),
            cfg["likelihood_threshold"])

    horizontal_diameter = euclidean_distance(right, left)
    vertical_diameter = euclidean_distance(ventral, dorsal)

    # Flag frame-to-frame jumps as tracking artifacts (same logic as the
    # original walkthrough script -- just vectorised via .mask instead of
    # boolean indexing in place).
    thresh = cfg["diameter_jump_threshold"]
    horizontal_diameter = horizontal_diameter.mask(horizontal_diameter.diff().abs() > thresh)
    vertical_diameter = vertical_diameter.mask(vertical_diameter.diff().abs() > thresh)

    pupil_area = np.pi * (horizontal_diameter / 2) * (vertical_diameter / 2)

    fs = cfg["fs"]
    target_fs = cfg["target_fs"]
    if fs / target_fs != round(fs / target_fs):
        LOGGER.warning(
            "fs/target_fs = %.3f is not an integer for %s; downsampling "
            "block size will be rounded -- check output alignment.",
            fs / target_fs, csv_path.name,
        )

    b0 = imaging_start + int(round(cfg["baseline_start_sec"] * fs))
    b1 = imaging_start + int(round(cfg["baseline_end_sec"] * fs))
    if b1 > len(pupil_area):
        raise ValueError(
            f"Baseline window [{b0}:{b1}] exceeds file length "
            f"({len(pupil_area)} frames)"
        )
    baseline = np.nanmean(pupil_area.iloc[b0:b1])
    if not np.isfinite(baseline) or baseline == 0:
        raise ValueError("Could not compute a valid (non-zero, finite) baseline")
    pupil_area_norm = pupil_area / baseline

    pupil_area_ds = trim_and_downsample(pupil_area_norm, imaging_start, fs, target_fs)
    n_out = len(pupil_area_ds)

    result = pd.DataFrame({
        "frame": np.arange(n_out),
        "time_sec": np.arange(n_out) / target_fs,
        "pupil_area_norm": pupil_area_ds,
    })

    diagnostics = dict(
        imaging_start_used=int(imaging_start),
        n_frames_raw=int(len(pupil_area)),
        n_frames_out=n_out,
        baseline_value=float(baseline),
        pct_nan_horizontal=float(horizontal_diameter.isna().mean() * 100),
        pct_nan_vertical=float(vertical_diameter.isna().mean() * 100),
    )

    # --- Optional: vertical eyelid aperture (dorsaleyelid <-> ventraleyelid) ---
    if cfg["compute_vertical_aperture"]:
        dorsal_eyelid_name = try_resolve_bodypart(available, cfg["dorsal_eyelid"], "dorsal_eyelid")
        ventral_eyelid_name = try_resolve_bodypart(available, cfg["ventral_eyelid"], "ventral_eyelid")
        if dorsal_eyelid_name and ventral_eyelid_name:
            dorsal_eyelid = extract_keypoint(data, dorsal_eyelid_name, coords=("x", "y"))
            ventral_eyelid = extract_keypoint(data, ventral_eyelid_name, coords=("x", "y"))
            if cfg["likelihood_threshold"] is not None:
                dorsal_eyelid = apply_likelihood_mask(
                    dorsal_eyelid,
                    extract_keypoint(data, dorsal_eyelid_name, coords=("likelihood",)),
                    cfg["likelihood_threshold"])
                ventral_eyelid = apply_likelihood_mask(
                    ventral_eyelid,
                    extract_keypoint(data, ventral_eyelid_name, coords=("likelihood",)),
                    cfg["likelihood_threshold"])
            vertical_aperture = euclidean_distance(ventral_eyelid, dorsal_eyelid)
            result["vertical_aperture_px"] = trim_and_downsample(
                vertical_aperture, imaging_start, fs, target_fs)
            diagnostics["pct_nan_vertical_aperture"] = float(vertical_aperture.isna().mean() * 100)
        else:
            LOGGER.warning(
                "Skipping vertical aperture for %s: eyelid keypoints not found "
                "(dorsal_eyelid=%r, ventral_eyelid=%r). Available bodyparts: %s",
                csv_path.name, cfg["dorsal_eyelid"], cfg["ventral_eyelid"], available,
            )

    # --- Optional: nose framewise displacement ---
    if cfg["compute_nose_motion"]:
        nose_name = try_resolve_bodypart(available, cfg["nose"], "nose")
        if nose_name:
            nose = extract_keypoint(data, nose_name, coords=("x", "y"))
            if cfg["likelihood_threshold"] is not None:
                nose = apply_likelihood_mask(
                    nose, extract_keypoint(data, nose_name, coords=("likelihood",)),
                    cfg["likelihood_threshold"])
            nose_fd = framewise_displacement(nose)
            result["nose_displacement_px"] = trim_and_downsample(
                nose_fd, imaging_start, fs, target_fs)
        else:
            LOGGER.warning(
                "Skipping nose motion for %s: nose keypoint not found (nose=%r). "
                "Available bodyparts: %s",
                csv_path.name, cfg["nose"], available,
            )

    return result, diagnostics


def save_diagnostic_plot(result: pd.DataFrame, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(result["time_sec"], result["pupil_area_norm"], linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalised pupil area")
    ax.set_title(title)
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
    logging.getLogger("matplotlib").setLevel(logging.WARNING)  # keep --verbose readable
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

    imaging_start_map = {}
    if cfg["imaging_start_file"]:
        try:
            imaging_start_map = load_imaging_start_map(cfg["imaging_start_file"])
        except (OSError, ValueError) as exc:
            LOGGER.error("Could not load --imaging-start-file %s: %s",
                         cfg["imaging_start_file"], exc)
            sys.exit(1)
        LOGGER.info("Loaded %d imaging_start entries from %s",
                    len(imaging_start_map), cfg["imaging_start_file"])

    summary_rows = []
    n_ok = n_fail = 0

    for csv_path in input_files:
        LOGGER.info("Processing %s", csv_path.name)
        try:
            imaging_start = resolve_imaging_start(csv_path, cfg, imaging_start_map)
            result, diag = process_pupil_file(csv_path, cfg, imaging_start)
            out_csv = output_dir / f"{csv_path.stem}_pupil_measures.csv"
            result.to_csv(out_csv, index=False)
            if cfg["save_plots"]:
                save_diagnostic_plot(
                    result, plot_dir / f"{csv_path.stem}_pupil_area.png", csv_path.stem
                )
            summary_rows.append(dict(file=csv_path.name, status="ok",
                                      output=str(out_csv), **diag))
            n_ok += 1
        except Exception as exc:  # keep batch alive on a per-file failure
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
