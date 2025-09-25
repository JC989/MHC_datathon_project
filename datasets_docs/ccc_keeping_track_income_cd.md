### Dataset: Keeping Track Online (CCC NYC) — Community District Income & Child Indicators

- **Portal**: `https://data.cccnewyork.org/`
- **Download hub**: `https://data.cccnewyork.org/data/download#0/66`
- **Geography**: Community District (CD) — align to CDTA codes for joins
- **Intended use**: Equity join for violation-rate gradient vs income/child indicators

#### Suggested fields to extract (normalize names)
- **CDTA Code** — `cdtacode` (e.g., BK01, MN12)
- **CDTA Name** — `cdtaname`
- **Year** — `year` (ACS vintage or KT reference year)
- **Median Household Income (USD)** — `median_household_income`
- **Child Poverty Rate (% <18 below poverty)** — `child_poverty_rate`
- (Optional) **Children <200% FPL (%)** — `children_below_200fpl_pct`
- (Optional) **Population <18** — `children_population`
- (Optional) **Total Population** — `total_population`

#### Join keys and workflow
- Join to analysis table via `cdtacode` using the equivalency file `hm78-6dwm` (Tract→NTA/CDTA) or directly when your unit is CDTA.
- If the CCC file uses names like "Brooklyn CD 1", derive `cdtacode` by mapping borough prefix (BK, BX, MN, QN, SI) + zero‑padded district number.

#### Notes
- Exact column names vary by export; standardize to the field list above during ingest.
- Prefer CDTA‑level metrics for consistent joins with NTAs via the equivalency crosswalk when needed.


