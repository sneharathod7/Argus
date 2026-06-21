# Data Preprocessing Report

## Summary
- **Initial Records:** 298,450
- **Final Records:** 235,128
- **Records Removed:** 63,322 (21.22%)

## Removal Breakdown
| Step | Records Removed | Reason |
|------|----------------|--------|
| Validation Status | 57,796 | Removed rejected/duplicate/processing/created1 |
| Near-Duplicate Dedup | 5,521 | Same vehicle, same grid, within 5 minutes |
| GPS Outliers | 0+ | Outside Bengaluru bbox or IQR outliers |

## New Features Added
| Feature | Description |
|---------|-------------|
| `geohash_7` | Geohash precision-7 spatial grid ID |
| `weighted_violation_score` | Number of offence codes per event (severity proxy) |
| `response_time_minutes` | Modified - Created datetime (enforcement speed) |
| `num_offence_codes` | Count of individual violation types per event |

## Data Quality Assessment
- **Weighted Violation Score:** mean=1.16, max=11
- **Response Time:** mean=499.8 min, median=15.9 min
- **Time Range:** 2023-11-09 19:11:46+00:00 to 2024-04-08 17:30:46+00:00
- **Unique Grids (Geohash-7):** 5,437
- **Unique Vehicles:** 191,828
