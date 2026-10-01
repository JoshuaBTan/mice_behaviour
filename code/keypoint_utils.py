# -*- coding: utf-8 -*-
"""
keypoint_utils.py

Shared helpers for reading the DeepLabCut-style tracking CSVs used
throughout this project and for pulling out per-keypoint x/y/likelihood
data from them.

Expected CSV layout (same as in pupil_analysis.py / body_analysis.py):

    row 0 : bodypart name, repeated for each of its x / y / likelihood
            columns
    row 1 : coordinate label ("x", "y", "likelihood")
    row 2+: numeric data, one row per frame

Column 0 is an index/scorer column and is ignored by these helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Loading / bodypart discovery
# ---------------------------------------------------------------------

def load_dlc_csv(path) -> pd.DataFrame:
    """Load a raw DLC-style CSV exactly as exported (no header parsing)."""
    return pd.read_csv(path)


def list_bodyparts(data: pd.DataFrame) -> list:
    """
    Return the sorted, unique bodypart names found in row 0 of `data`
    (excluding the first, index/scorer column).
    """
    return sorted(set(data.iloc[0].values[1:].tolist()))


def extract_keypoint(data: pd.DataFrame, bodypart: str,
                      coords=("x", "y")) -> pd.DataFrame:
    """
    Pull out a (n_frames, len(coords)) float DataFrame for one bodypart.

    Parameters
    ----------
    data     : raw DataFrame as returned by load_dlc_csv
    bodypart : exact bodypart name (row 0 label) to extract
    coords   : which coordinate columns to keep, e.g. ("x", "y") or
               ("likelihood",)
    """
    mask = (data.iloc[0] == bodypart) & (data.iloc[1].isin(coords))
    if not mask.any():
        raise ValueError(
            f"No columns found for bodypart '{bodypart}' with coords {coords}"
        )
    temp = data.loc[2:, mask].astype(float).reset_index(drop=True)
    temp.columns = data.loc[1, mask].values
    # keep columns in the order requested, dropping any coord not present
    ordered = [c for c in coords if c in temp.columns]
    return temp[ordered]


def try_resolve_bodypart(available: list, requested: str, role_label: str = ""):
    """
    Same resolution logic as resolve_bodypart, but returns None instead of
    raising when the keypoint can't be found or is ambiguous. Useful for
    optional/supplementary measures where a missing keypoint should just
    disable that one measure rather than fail the whole file.
    """
    try:
        return resolve_bodypart(available, requested, role_label)
    except ValueError:
        return None


def resolve_bodypart(available: list, requested: str, role_label: str = "") -> str:
    """
    Resolve a user-supplied keypoint name/alias against the bodyparts
    actually present in a file.

    Tries, in order: exact match, case-insensitive exact match, then a
    unique case-insensitive substring match. Raises a clear error if the
    keypoint can't be found or the substring match is ambiguous, so
    mistakes surface immediately instead of silently pulling the wrong
    column.
    """
    if requested in available:
        return requested

    lowered = {a.lower(): a for a in available}
    if requested.lower() in lowered:
        return lowered[requested.lower()]

    matches = [a for a in available if requested.lower() in a.lower()]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        raise ValueError(
            f"Ambiguous keypoint alias '{requested}' for role '{role_label}': "
            f"matches {matches}. Set the exact name in your config/CLI args."
        )
    else:
        raise ValueError(
            f"Could not find keypoint '{requested}' for role '{role_label}'. "
            f"Available bodyparts in this file: {available}"
        )


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------

def euclidean_distance(p1: pd.DataFrame, p2: pd.DataFrame) -> pd.Series:
    """Frame-wise Euclidean distance between two (x, y) DataFrames."""
    dx = p1["x"] - p2["x"]
    dy = p1["y"] - p2["y"]
    return np.sqrt(dx ** 2 + dy ** 2)


def framewise_displacement(p: pd.DataFrame) -> pd.Series:
    """
    Frame-to-frame Euclidean displacement of a single (x, y) keypoint.
    First sample is NaN (nothing to diff against) so the output stays
    aligned to the original frame index.
    """
    dx = p["x"].diff()
    dy = p["y"].diff()
    return np.sqrt(dx ** 2 + dy ** 2)


def angle_deg(p_from: pd.DataFrame, p_to: pd.DataFrame) -> np.ndarray:
    """Angle in degrees of the vector p_to - p_from, per frame."""
    dx = p_to["x"] - p_from["x"]
    dy = p_to["y"] - p_from["y"]
    return np.degrees(np.arctan2(dy, dx))


def angular_velocity_deg(angle_degrees: np.ndarray, fs: float) -> np.ndarray:
    """
    Angular velocity in degrees/s from an angle (degrees) trace, unwrapping
    to avoid +/-180 degree jumps. Output is padded with a leading NaN so
    it stays the same length as the input (aligned to frame index).
    """
    angle_rad = np.radians(angle_degrees)
    unwrapped = np.unwrap(angle_rad)
    vel = np.degrees(np.diff(unwrapped) * fs)
    return np.concatenate([[np.nan], vel])


def circular_mean_deg(angles_deg: np.ndarray) -> float:
    """
    Circular mean of an angle sample (degrees), ignoring NaNs.

    A plain arithmetic mean is wrong for angles: e.g. mean(179, -179) should
    be 180 (they're 2 degrees apart across the wrap), not 0 (which a plain
    mean would give). This averages on the unit circle instead
    (mean of sin/cos components, then atan2 back to degrees) so wraparound
    is handled correctly.
    """
    angles_deg = np.asarray(angles_deg, dtype=float)
    rad = np.radians(angles_deg)
    sin_mean = np.nanmean(np.sin(rad))
    cos_mean = np.nanmean(np.cos(rad))
    if np.isnan(sin_mean) or np.isnan(cos_mean):
        return np.nan
    return float(np.degrees(np.arctan2(sin_mean, cos_mean)))


def trim_and_downsample(series: pd.Series, start_frame: int, fs: float,
                         target_fs: float, circular: bool = False) -> np.ndarray:
    """
    Trim a native-fs series to start at `start_frame`, then block-average
    down to `target_fs` (e.g. 30 Hz camera -> 10 Hz imaging). Trailing
    frames that don't fill a complete block are dropped. NaNs within a
    block are ignored when averaging (pandas default skipna behaviour).

    Parameters
    ----------
    circular : if True, average each block with circular_mean_deg instead
               of a plain mean -- use this for angle-in-degrees columns
               (e.g. body_angle_deg, tail_angle_deg) that can wrap at
               +/-180 degrees. Leave False for everything else (positions,
               displacements, angular velocity, lengths, etc.).
    """
    trimmed = pd.Series(series).iloc[start_frame:].reset_index(drop=True)
    factor = max(1, int(round(fs / target_fs)))
    n_complete = len(trimmed) // factor
    trimmed = trimmed.iloc[: n_complete * factor]
    groups = trimmed.groupby(np.arange(len(trimmed)) // factor)
    if circular:
        return groups.apply(lambda g: circular_mean_deg(g.values)).reset_index(drop=True).values
    return groups.mean().reset_index(drop=True).values


def apply_likelihood_mask(xy_df: pd.DataFrame, likelihood_df: pd.DataFrame,
                           threshold: float) -> pd.DataFrame:
    """Set (x, y) to NaN on any frame where likelihood < threshold."""
    if threshold is None:
        return xy_df
    out = xy_df.copy()
    bad = (likelihood_df["likelihood"] < threshold).values
    out.loc[bad, :] = np.nan
    return out
