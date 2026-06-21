# Dense Spatiotemporal Panel Report

## Panel Structure
- **Total Rows:** 1,127,351
- **Dense Panel (Top-300 Grids):** 1,086,900 rows (complete hourly timeline)
- **Sparse Events (Remaining Grids):** 40,451 rows (event-driven)
- **Coverage:** Top-300 grids cover 66.2% of all violations
- **Time Range:** 2023-11-09 19:00:00+00:00 -> 2024-04-08 17:00:00+00:00 (3,623 hours)

## Sparsity Analysis (Dense Panel Only)
- **Non-zero hours:** 86,203
- **Zero-fill hours:** 1,041,148
- **Fill rate in dense panel:** 7.93%

## New Features per Grid-Hour
| Feature | Description |
|---------|-------------|
| `violation_count` | Raw event count |
| `weighted_violation_count` | Sum of offence codes (severity-weighted) |
| `unique_vehicle_count` | Distinct vehicles |
| `heavy_vehicle_ratio` | Fraction of trucks/buses/goods vehicles |
| `multi_violation_ratio` | Fraction of events with 2+ offence codes |
| `violation_entropy` | Shannon entropy of violation types |
| `num_devices_active` | Number of monitoring devices active |
| `repeat_offender_ratio` | Fraction of vehicles previously seen in this grid |
| `avg_response_time` | Mean enforcement response time (minutes) |
| `grid_lat`, `grid_lng` | Grid centroid coordinates |
| `is_dense_grid` | Whether this grid has full hourly materialization |
