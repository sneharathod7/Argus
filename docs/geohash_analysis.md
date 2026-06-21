# Geohash Spatio-Temporal Analysis

## 1. Geohash Precision Evaluation

We evaluated the entire dataset across two Geohash precisions:

| Precision | Dimensions (Approx) | Unique Cells | Avg Violations / Cell |
|-----------|--------------------|--------------|-----------------------|
| **Geohash 7** | ~153m x 153m | 5,753 | 51.88 |
| **Geohash 8** | ~38m x 19m | 28,149 | 10.60 |

### Recommendation: Geohash-7
**Geohash-7 is the vastly superior operational granularity.** 
Geohash-8 creates severe spatial sparsity (28,149 unique tiny cells). Since our timeline is binned hourly, using Geohash-8 would result in a massive, sparse matrix where the majority of cells have 0 violations per hour. Geohash-8 acts effectively as a single-point cluster rather than a geographic grid.

Geohash-7 represents roughly 1-2 city blocks, which is perfectly aligned with operational patrols and provides enough density for a robust forecasting time-series signal. The canonical dataset `grid_hourly_table.parquet` has been built using Geohash-7.

## 2. Spatio-Temporal Grid Statistics (Geohash-7)

- **Total Active Grids:** 5,753
- **Dataset Time Range:** 3,623 hours
- **Total Possible Grid-Hours:** 20,843,119
- **Populated Grid-Hours:** 99,140
- **Temporal Sparsity:** 99.52% (Percentage of theoretical maximum Grid-Hours with zero violations)

### Top 10 Grids by Total Violation Volume
- **tdr1v21**: 5,220 violations
- **tdr1v63**: 3,319 violations
- **tdr1y54**: 3,257 violations
- **tdr1u9f**: 2,967 violations
- **tdr1xfs**: 2,871 violations
- **tdr1y51**: 2,787 violations
- **tdr1v66**: 2,728 violations
- **tdr1y50**: 2,667 violations
- **tdr1vgt**: 2,655 violations
- **tdr1v3e**: 2,363 violations
