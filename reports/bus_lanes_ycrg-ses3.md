### Dataset: Bus Lanes – Local Streets (Map)

- **Source URL**: `https://data.cityofnewyork.us/resource/ycrg-ses3.json`
- **Approx. Rows**: 3,783
- **Columns**: 29

#### Columns and API fields
- **Geometry** — API: `the_geom` — Type: MultiLineString (WGS84)
- **Street** — API: `street` — Type: Text
- **Bus Lane Traffic Direction** — API: `bltrafdir` — Type: Text (e.g., T)
- **Segment ID** — API: `segmentid` — Type: Text
- **Roadway Type** — API: `rw_type` — Type: Text
- **Street Width** — API: `streetwidt` — Type: Number
- **Borough** — API: `boro` — Type: Text (BK, BX, MN, QN, SI)
- **Facility / Corridor** — API: `facility` — Type: Text
- **Direction** — API: `direction` — Type: Text (NB, SB, EB, WB)
- **Hours** — API: `hours` — Type: Text (e.g., 7AM-10AM)
- **Days** — API: `days` — Type: Text (e.g., Mon - Fri, 7 Days/Week)
- **Days Code** — API: `days_code` — Type: Number
- **Lane Width (label)** — API: `lane_width` — Type: Text (Single/Double)
- **Lane Type 1** — API: `lane_type1` — Type: Text (e.g., Bus Lane, Shared Lane)
- **Lane Type** — API: `lane_type` — Type: Text (e.g., Curbside, Offset)
- **Lane Color** — API: `lane_color` — Type: Text (e.g., Red)
- **SBS Route 1** — API: `sbs_route1` — Type: Text (e.g., B46)
- **Open Dates** — API: `open_dates` — Type: Date (MM/DD/YYYY)
- **Year 1** — API: `year1` — Type: Number
- **Year 2** — API: `year2` — Type: Number
- **Year 3** — API: `year3` — Type: Number
- **Last Update** — API: `last_updat` — Type: Date/Text
- **Chron ID** — API: `chron_id_1` — Type: Text
- **Shape Length (degrees)** — API: `shape_leng` — Type: Number
- **Shape Length (state plane)** — API: `shape_le_1` — Type: MultiLineString (NY State Plane feet)

#### Notes
- Use `the_geom` for mapping and length; for accurate lane-miles, project to EPSG:2263 and sum segment lengths by area.
- `lane_type`/`lane_type1` distinguish curbside/offset and bus lane/shared segments; filter to true bus lanes when building denominators.
- Time-based restrictions are encoded in `hours` and `days`.


