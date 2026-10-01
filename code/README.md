# Behaviour preprocessing: batch pupil & body analysis

Automated, config-driven versions of `pupil_analysis.py` and
`body_analysis.py`. Instead of hand-editing a path and re-running per
video, point these at a folder of DeepLabCut-style tracking CSVs and they
process every matching file in one command.

## Files

| File                        | Purpose                                              |
|------------------------------|-------------------------------------------------------|
| `keypoint_utils.py`          | Shared helpers: load/parse tracking CSVs, resolve keypoint names, geometry (distance, displacement, angle, angular velocity), trim+downsample |
| `batch_pupil_analysis.py`    | CLI script -> normalised pupil area, eyelid aperture, nose motion |
| `batch_body_analysis.py`     | CLI script -> back centroid, body angle, body length, hindlimb & tail kinematics |
| `pupil_config.yaml`          | Example/editable config for the pupil script          |
| `body_config.yaml`           | Example/editable config for the body script            |
| `imaging_start_times.example.json` | Example per-video imaging-start-frame map for the pupil script |
| `requirements.txt`           | `pip install -r requirements.txt`                      |

Both scripts expect the same 3-row-header tracking CSV layout used
throughout the project (bodypart / coord / data), i.e. what you get from
`pd.read_csv(...)` on a DLC-style export:

```
row 0 (after header) : bodypart name, repeated per x/y/likelihood column
row 1                : "x" / "y" / "likelihood"
row 2+                : numeric data, one row per frame
```

## Install

```bash
pip install -r requirements.txt
```

PyYAML is only needed if you want to use `.yaml`/`.yml` configs; JSON
configs (`.json`) work without it.

## Quick start

```bash
# Pupil
python batch_pupil_analysis.py \
    --input-dir /path/to/pupil_data/sub-waves04/video_preds \
    --output-dir ./pupil_output \
    --fs 30 --target-fs 10 --imaging-start 66

# Body
python batch_body_analysis.py \
    --input-dir /path/to/body_data/sub-waves04/video_preds \
    --output-dir ./body_output \
    --fs 30 --target-fs 10
```

Or via config file (copy + edit `pupil_config.yaml` / `body_config.yaml`,
then any CLI flag you also pass overrides the matching config value):

```bash
python batch_pupil_analysis.py --config pupil_config.yaml
python batch_pupil_analysis.py --config pupil_config.yaml --target-fs 20   # one-off override
```

## What each script computes

### `batch_pupil_analysis.py`
Per file, mirrors the walkthrough script:
- horizontal/vertical pupil diameter (left/right, dorsal/ventral pupil
  keypoints), with frame-to-frame jumps above `diameter_jump_threshold`
  flagged as tracking artifacts (NaN)
- pupil area, normalised to a baseline window (`baseline_start_sec` -
  `baseline_end_sec` after `imaging_start`)
- trimmed to `imaging_start` and block-averaged from `fs` down to
  `target_fs`

**Per-video imaging start:** since the LED-flash frame (imaging start)
typically differs per video, `imaging_start` doesn't have to be one fixed
number for the whole batch. Set `--imaging-start-file path/to/map.json`
(or `imaging_start_file:` in the config) to a JSON file mapping each
video's file stem (or full filename) to its own start frame:

```json
{
  "sub-waves04_ses-3_task-rest_run-1_face": 66,
  "sub-waves04_ses-3_task-rest_run-2_face": 82
}
```

See `imaging_start_times.example.json` for a template. Resolution order
per file: (1) an entry in `imaging_start_file` for that file's stem, then
its full filename, (2) `--imaging-start` as a fallback default if no entry
is found. If neither resolves for a given file, that one file fails
(logged in `run_summary.csv` with the reason) instead of silently using
the wrong frame -- everything else in the batch still runs. The frame
actually used for each file is recorded as `imaging_start_used` in
`run_summary.csv`, so you can double check nothing fell back unexpectedly.

Plus two **optional** supplementary measures picked up from the keypoints
you added (`dorsaleyelid`/`ventraleyelid`, `nose`):
- `vertical_aperture_px` — distance between dorsal/ventral eyelid, same
  trim+downsample treatment as pupil area
- `nose_displacement_px` — frame-to-frame nose movement, same treatment

These are resolved **leniently**: if a file doesn't have the eyelid or
nose keypoints (e.g. an older recording), that one column is just skipped
for that file (logged as a warning) instead of failing the whole file.
Disable either with `--skip-vertical-aperture` / `--skip-nose-motion` if
you never want them computed.

Output columns: `frame, time_sec, pupil_area_norm, vertical_aperture_px, nose_displacement_px`

### `batch_body_analysis.py`
Per file, computed at the tracking data's **native** frame rate (`fs`),
then **block-averaged down to `target_fs`** (same treatment as
`pupil_area_norm` in the pupil script):
- back centroid (midpoint of `back_mid`/`back_low`) and its frame-to-frame
  displacement
- body angle (`back_mid` -> `back_low`) and angular velocity
- body length (`back_low` to `back_mid`) and its rate of change
- hindlimb (`hind_left`, `hind_right`) frame-to-frame displacement
- tail angle (`tail_base` -> `tail_mid`) and angular velocity

**Downsampling and angle wraparound:** `body_angle_deg` and
`tail_angle_deg` are averaged within each block using a **circular mean**
(correct across the +/-180 degree wraparound — a plain mean of e.g. 179°
and -179° would wrongly give ~0° instead of ~180°). Every other column
(positions, displacements, angular velocity, body length) uses a normal
mean of its native-fs values within the block.

**If you used an earlier version of this script:** `fs` used to mean "the
rate the data is already at" and was used directly for angular velocity
with **no downsampling step** — implicitly assuming pre-downsampled 10 Hz
data (default was `fs=10`). It now means the tracking data's **native**
rate (default `fs=30`, matching the camera), with a separate `target_fs`
(default `10`) controlling the block-averaging. If your tracking really is
native 10 Hz with nothing to downsample, pass `--fs 10 --target-fs 10` to
reproduce the old, non-downsampling behaviour.

**Note on units:** the original walkthrough computed body angular velocity
in rad/s but tail angular velocity in deg/s (an inconsistency in the
original script). This batch version reports **both in deg/s** for
consistency — keep that in mind if you're comparing directly against
numbers from the old script.

Output columns: `frame, time_sec, back_center_x, back_center_y, back_center_fd, body_angle_deg, body_angular_velocity_degs, body_length_px, body_length_dt, hind_left_fd, hind_right_fd, tail_angle_deg, tail_angular_velocity_degs`

## Keypoint handling

Neither script hard-codes exact DLC label strings. `keypoint_utils.list_bodyparts`
auto-detects whatever bodyparts are in a given file (same approach as
`body_analysis.py`), and `resolve_bodypart` matches your configured
name/alias against them (exact match -> case-insensitive match -> unique
substring match), raising a clear error listing the available bodyparts if
it can't find or disambiguate one. So if your labelling scheme differs
from file to file or across cohorts, you generally don't need per-file
edits — just set the role names once in the config (or leave the
defaults, which match what's used in the walkthrough scripts).

## Batch behaviour

- Every file matching `--pattern` under `--input-dir` (optionally
  `--recursive`) is processed independently; a failure on one file (bad
  baseline window, missing required keypoint, corrupt CSV, etc.) is
  logged and the run continues with the rest.
- A `run_summary.csv` is always written to `--output-dir`, one row per
  input file, with `status` (`ok`/`failed`), the output path or error
  message, and basic per-file diagnostics (NaN rates, baseline value,
  frame counts, etc.) so you can spot problem files without re-running
  everything.
- `--save-plots` writes one diagnostic PNG per file under
  `<output_dir>/plots/` (pupil area trace, or body angle / centroid /
  hindlimb displacement) — useful for a first pass, but off by default
  since it adds time for large batches.
- `--verbose` turns on debug logging (resolved config, etc.).

## Extending

Both `_load_role_xy`-style keypoint extraction and the `keypoint_utils`
helpers (`euclidean_distance`, `framewise_displacement`, `angle_deg`,
`angular_velocity_deg`, `trim_and_downsample`) are reusable if you want to
add another measure later (e.g. a third eyelid distance, or an additional
body angle) — plug the keypoint role into the config/CLI args the same way
`nose`/`dorsal_eyelid`/`back_low` etc. are done, and call the relevant
helper in `process_pupil_file` / `process_body_file`.

The original QC step (boxplot of `predictions_pixel_error.csv` model
prediction error, train vs. validation) wasn't batched here since it
already operates on one aggregate file across all videos rather than
per-video tracking CSVs — happy to add a batch/CLI version of that too if
useful.
