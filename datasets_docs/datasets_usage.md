### Datasets and how we'll use them

- **ACE Violations (kh8p-hcbm)** — `https://data.ny.gov/resource/kh8p-hcbm.json`
  - Points + timestamps → hotspot mapping and hour-of-day patterns
  - Use: `violation_georeference` (or `violation_latitude/longitude`), `first_occurrence`, `violation_type`, `violation_status`, `bus_route_id`, `stop_id`
  - Derive: `hour_local` (ET), near-POI flags (schools/hospitals), CBD flag (optional)

- **ACE Enforced Routes (ki2b-sg5y)** — `https://data.ny.gov/resource/ki2b-sg5y.json`
  - Implementation dates; context of where/when ABLE→ACE
  - Join: `bus_route_id` (violations) ↔ `route`; fields: `program`, `implementation_date`

- **Bus Route Segment Speeds 2025 (kufs-yh3x)** — `https://data.ny.gov/resource/kufs-yh3x.json`
  - Hourly speeds/travel time between timepoint stop pairs
  - Optional linkage to show delay at hotspots; summarize `average_road_speed`, `average_travel_time`, `bus_trip_count` by `route_id/direction/hour_of_day`

- **Bus Route Segment Speeds 2023–2024 (58t6-89vi)** — `https://data.ny.gov/resource/58t6-89vi.json`
  - Historical comparison/baseline to 2025; same fields/usage as above

- **Bus Lanes – Local Streets (ycrg-ses3)** — `https://data.cityofnewyork.us/resource/ycrg-ses3.json`
  - Geometry → exposure denominator: lane‑miles by area
  - Steps: filter true bus lanes (e.g., `lane_type1='Bus Lane'`), project to EPSG:2263, compute segment length (miles), spatial intersect with NTA/CDTA, sum to `lane_miles`
  - Alternate source id: `rx8t-6euq`

- **Neighborhood Tabulation Areas 2020 (9nt8-h7nd)** — `https://data.cityofnewyork.us/resource/9nt8-h7nd.json`
  - Geography for equity analysis; spatial join violations to `nta2020`

- **Community District Tabulation Areas 2020 (xn3r-zk6y)** — `https://data.cityofnewyork.us/resource/xn3r-zk6y.json`
  - Alternative geography keyed to CD (aligns with CCC income); optional rollup

- **2020 Census Tracts → NTA/CDTA Equivalency (hm78-6dwm)** — `https://data.cityofnewyork.us/resource/hm78-6dwm.json`
  - Bridge ACS/CCC indicators to analysis geographies; keys: `geoid` ↔ tract datasets, `ntacode`, `cdtacode`

- **NYC Schools (s3k6-pzi2)** — `https://data.cityofnewyork.us/resource/s3k6-pzi2.json`
  - Point locations for DOE schools; proximity analysis (buffer 100 m)
  - Filter to Manhattan via `borough in ('M','Manhattan')`; use `location_1` for geometry

- **NYC Health + Hospitals Facilities (q6fj-vxf8)** — `https://data.cityofnewyork.us/resource/q6fj-vxf8.json`
  - Point locations (Acute Care, Child Health Centers, etc.); proximity analysis (buffer 100 m)
  - Filter to Manhattan with `borough='Manhattan'`; use `location_1` for geometry

- **Keeping Track Online (CCC NYC) — Income/Child indicators (CD level)**
  - Portal: `https://data.cccnewyork.org/` (Download hub: `https://data.cccnewyork.org/data/download#0/66`)
  - Use: `median_household_income`, `child_poverty_rate` (normalize to `cdtacode`), create income deciles; join to CDTA or via equivalency

### Core metrics
- Violation rate (KPI): `rate = violations / lane_miles` by area × hour
- Equity gradient: rate vs income decile (CDTA or NTA mapped to CDTA)
- School/hospital proximity spikes: compare specified hour windows vs other hours

### API documentation quick links
- Pattern: `https://dev.socrata.com/foundry/{portal}/{dataset_id}`
  - NYC Open Data portal: `data.cityofnewyork.us`
  - NY State (MTA) portal: `data.ny.gov`
- Examples:
  - ACE Violations (`kh8p-hcbm`): `https://dev.socrata.com/foundry/data.ny.gov/kh8p-hcbm`
  - ACE Enforced Routes (`ki2b-sg5y`): `https://dev.socrata.com/foundry/data.ny.gov/ki2b-sg5y`
  - Bus Route Segment Speeds 2025 (`kufs-yh3x`): `https://dev.socrata.com/foundry/data.ny.gov/kufs-yh3x`
  - Bus Lanes – Local Streets (`ycrg-ses3`): `https://dev.socrata.com/foundry/data.cityofnewyork.us/ycrg-ses3`
  - NTAs 2020 (`9nt8-h7nd`): `https://dev.socrata.com/foundry/data.cityofnewyork.us/9nt8-h7nd`
  - Tract→NTA/CDTA Equivalency (`hm78-6dwm`): `https://dev.socrata.com/foundry/data.cityofnewyork.us/hm78-6dwm`

### API limits and paging
- Default response size is 1,000 rows. Increase with `?$limit=` (typical max 50,000) and paginate with `?$offset=`.
- Always include an `$order` for stable paging.
- Examples:
  - First 50k: `...?$limit=50000`
  - Next 50k: `...?$limit=50000&$offset=50000`
  - Count only: `...?$select=count(*)`
- For higher throughput, use an app token header `X-App-Token`.
- Docs: [Socrata limit](https://dev.socrata.com/docs/queries/limit.html), [offset](https://dev.socrata.com/docs/queries/offset.html).


