# Race Shape Sidecar

## Purpose

The Race Shape sidecar captures race-execution signals that are not represented by standard form, ratings, odds, or result-only data. It is designed as a shadow-only analytics layer: it can parse archived profile HTML, build run-level records, and aggregate jockey execution-variance profiles, but it does not alter live VÉLØ scoring.

## Current Boundary

Live scoring is not touched.

The current implementation only adds:

- `src/race_shape/form_history_parser.py` — archived form table parser.
- `src/race_shape/models.py` — portable dataclasses for run records and jockey profiles.
- `src/race_shape/ride_profile.py` — aggregation logic for jockey profile artifacts.
- `scripts/ops/build_race_shape_profiles.py` — CLI for generating JSON artifacts under `data/race_shape`.
- `tests/test_race_shape_sidecar.py` — regression coverage for parser and profile builder behavior.

## Data Flow

```text
Archived RP-style profile HTML
  -> RpFormHistoryParser
  -> RaceShapeRun[]
  -> JockeyRideProfileBuilder
  -> race_shape_runs.json + jockey_ride_profiles.json + build_metadata.json
```

## What This Measures

The sidecar uses market expectation versus result as an early validation proxy. A run is considered well-fancied when decimal SP is less than or equal to 4.0. A profile records whether those well-fancied runners finished outside an expected contention band.

This is not a live decision signal yet. It is a research artifact that needs sample-size validation, source-quality checks, and cross-validation with comments/video before promotion.

## Operational Usage

```bash
python scripts/ops/build_race_shape_profiles.py \
  --input-dir data/racing_post_account_raw/2026-05-26 \
  --output-dir data/race_shape/2026-05-26 \
  --min-jockey-rides 3
```

## Promotion Rules

Before this sidecar can influence VÉLØ scoring, it needs:

1. Coverage reports across multiple race days.
2. False-positive review against race comments and video.
3. Stable entity resolution for jockey, horse, course, and race identifiers.
4. Feature flags that default to off.
5. Backtests proving that the signal improves calibration without degrading ROI discipline.

## Architecture Rationale

This is deliberately implemented as a clean sidecar rather than a modification to `VeloPrime` or the Five-Filter System. The repository currently describes a modular ML/data architecture in the README, but the active agent remains a direct in-process orchestration class. Keeping Race Shape isolated prevents research code from contaminating production outputs while the evidence base is still being built.
