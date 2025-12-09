# VELO Training Data

**Last Updated**: December 9, 2025

## Files

### 1. raceform.csv (633MB, 1.7M rows)
Historical race results from 2015-2023.

**Columns**: date, course, race_id, off, race_name, type, class, pattern, rating_band, age_band, sex_rest, dist, going, ran, num, pos, draw, ovr_btn, btn, horse, age, sex, wgt, hg, time, sp, jockey, trainer, prize, or, rpr, ts, sire, dam, damsire, owner, comment

### 2. recent.csv (93MB, 254K rows)
Recent race results from 2024-present.

**Same schema as raceform.csv**

### 3. DataDescription_.xlsx
Data dictionary and column descriptions.

### 4. VELO_Oracle_2025-10-21_BACKFILL_v1.xlsx
Backfill metadata and validation.

## Total Dataset
- **Rows**: 1,956,501 races
- **Size**: 726MB
- **Date Range**: 2015-01-01 to 2024-12-09
- **Coverage**: UK & Ireland racing

## Usage

### Train VELO v11
```bash
python scripts/velo_offline_train.py \
  --data data/raceform.csv \
  --output-dir ./models \
  --seq-len 5
```

### Combine datasets
```bash
cat data/raceform.csv data/recent.csv > data/full_dataset.csv
```

## DO NOT DELETE THIS DATA
This is the permanent VELO training dataset. Stored in 3 locations:
1. GitHub: velo-oracle repo
2. Supabase: velo_training_data table
3. Local: /home/ubuntu/velo_data_backup.tar.gz
