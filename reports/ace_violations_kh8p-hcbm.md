### Dataset: MTA Bus Automated Camera Enforcement Violations (Beginning Oct 2019)

- **Source URL**: `https://data.ny.gov/resource/kh8p-hcbm.json`
- **Approx. Rows**: 3.78M
- **Columns**: 15

#### Columns and API fields
- **Violation ID** — API: `violation_id` — Type: Text (identifier)
- **Stop ID** — API: `stop_id` — Type: Text
- **Stop Name** — API: `stop_name` — Type: Text
- **Bus Stop Latitude** — API: `bus_stop_latitude` — Type: Number
- **Bus Stop Longitude** — API: `bus_stop_longitude` — Type: Number
- **Violation Georeference** — API: `violation_georeference` — Type: Point
- **Bus Stop Georeference** — API: `bus_stop_georeference` — Type: Point
- **Vehicle ID** — API: `vehicle_id` — Type: Text (hashed plate)
- **First Occurrence** — API: `first_occurrence` — Type: Timestamp
- **Last Occurrence** — API: `last_occurrence` — Type: Timestamp
- **Violation Status** — API: `violation_status` — Type: Text
- **Violation Type** — API: `violation_type` — Type: Text
- **Bus Route ID** — API: `bus_route_id` — Type: Text
- **Violation Latitude** — API: `violation_latitude` — Type: Number
- **Violation Longitude** — API: `violation_longitude` — Type: Number

#### Notes
- Use `first_occurrence` for hour-of-day features; `last_occurrence` is present for two-bus captures.
- Prefer geometry from `violation_georeference` when present; fall back to (`violation_latitude`, `violation_longitude`).
- Join to stops via `stop_id` and to routes via `bus_route_id` if needed.


