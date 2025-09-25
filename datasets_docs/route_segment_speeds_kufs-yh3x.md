### Dataset: MTA Bus Route Segment Speeds (Beginning 2025)

- **Source URL**: `https://data.ny.gov/resource/kufs-yh3x.json`
- **Approx. Rows**: 3.41M
- **Columns**: 24

#### Columns and API fields
- **Year** — API: `year` — Type: Number
- **Month** — API: `month` — Type: Number
- **Timestamp** — API: `timestamp` — Type: Floating Timestamp (MM/DD/YYYY HH:MM:SS)
- **Day of Week** — API: `day_of_week` — Type: Text
- **Hour of Day** — API: `hour_of_day` — Type: Number (0–23)
- **Route ID** — API: `route_id` — Type: Text
- **Direction** — API: `direction` — Type: Text (N, S, E, W)
- **Borough** — API: `borough` — Type: Text
- **Route Type** — API: `route_type` — Type: Text (Express, Local, Limited, SBS, School)
- **Stop Order** — API: `stop_order` — Type: Number
- **Timepoint Stop ID** — API: `timepoint_stop_id` — Type: Number
- **Timepoint Stop Name** — API: `timepoint_stop_name` — Type: Text
- **Timepoint Stop Latitude** — API: `timepoint_stop_latitude` — Type: Number
- **Timepoint Stop Longitude** — API: `timepoint_stop_longitude` — Type: Number
- **Next Timepoint Stop ID** — API: `next_timepoint_stop_id` — Type: Number
- **Next Timepoint Stop Name** — API: `next_timepoint_stop_name` — Type: Text
- **Next Timepoint Stop Latitude** — API: `next_timepoint_stop_latitude` — Type: Number
- **Next Timepoint Stop Longitude** — API: `next_timepoint_stop_longitude` — Type: Number
- **Road Distance** — API: `road_distance` — Type: Number (miles)
- **Average Travel Time** — API: `average_travel_time` — Type: Number (minutes)
- **Average Road Speed** — API: `average_road_speed` — Type: Number (mph)
- **Bus Trip Count** — API: `bus_trip_count` — Type: Number
- **Timepoint Stop Georeference** — API: `timepoint_stop_georeference` — Type: Point
- **Next Timepoint Stop Georeference** — API: `next_timepoint_stop_georeference` — Type: Point

#### Notes
- Segment metrics represent movement between `timepoint_stop_*` and `next_timepoint_stop_*` for a given hour, direction, and route.
- Speed is derived from distance and average travel time; counts provide reliability context.


