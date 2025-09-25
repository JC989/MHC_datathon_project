### Dataset: 2020 Neighborhood Tabulation Areas (NTAs)

- **Source URL**: `https://data.cityofnewyork.us/resource/9nt8-h7nd.json`
- **Approx. Rows**: 262
- **Columns**: 12
- **Row entity**: Neighborhood Tabulation Area (NTA)

#### Columns and API fields
- **BoroCode** — API: `borocode` — Type: Number
- **BoroName** — API: `boroname` — Type: Text
- **CountyFIPS** — API: `countyfips` — Type: Text
- **NTA2020** — API: `nta2020` — Type: Text (NTA code)
- **NTAName** — API: `ntaname` — Type: Text
- **NTAAbbrev** — API: `ntaabbrev` — Type: Text
- **NTAType** — API: `ntatype` — Type: Text (0=residential; non‑residential/special elsewhere)
- **CDTA2020** — API: `cdta2020` — Type: Text (Community District Tabulation Area code)
- **CDTAName** — API: `cdtaname` — Type: Text
- **Shape_Length** — API: `shape_leng` — Type: Number (internal units)
- **Shape_Area** — API: `shape_area` — Type: Number (internal units squared)
- **Geometry** — API: `the_geom` — Type: MultiPolygon (WGS84)

#### Notes
- Use `nta2020` as primary key when aggregating metrics by neighborhood; `cdta2020` enables crosswalk to CDTAs.
- For spatial analysis, project `the_geom` to EPSG:2263 to compute area/length accurately.


