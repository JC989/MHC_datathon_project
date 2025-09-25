### Dataset: MTA Bus Automated Camera Enforced Routes (Beginning Oct 2019)

- **Source URL**: `https://data.ny.gov/resource/ki2b-sg5y.json`
- **Approx. Rows**: 64
- **Columns**: 3

#### Columns and API fields
- **Route** — API: `route` — Type: Text
- **Program** — API: `program` — Type: Text (ABLE or ACE)
- **Implementation Date** — API: `implementation_date` — Type: Floating Timestamp (MM/DD/YYYY)

#### Notes
- Use to determine when ABLE vs ACE took effect by route; join to violations on `bus_route_id` ↔ `route`.


