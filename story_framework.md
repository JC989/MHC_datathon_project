# 5-Step Data Story Framework
*Team: Mohamed Hiba, Jean Carlo Chajon, Shazadul Islam*

## 1. Context – Why this story matters
New York City uses Automated Camera Enforcement (ACE) to keep bus lanes clear. But raw violation counts can be misleading: areas with more bus lane inevitably see more tickets. The question that matters for equitable, effective operations is: where and when are ACE violation rates (per bus‑lane mile) truly high, and how do these hotspots align with sensitive places like schools and hospitals? We focus on Manhattan to produce actionable guidance for enforcement and curb management.

## 2. Data – What’s powering your story
- ACE Violations (`kh8p-hcbm`) – points + timestamps
- Bus Lanes – Local Streets (`ycrg-ses3`) – line geometry → lane‑miles denominator
- NTAs 2020 (`9nt8-h7nd`) – neighborhood polygons
- Schools (DOE) (`s3k6-pzi2`) and NYC Health + Hospitals (`q6fj-vxf8`) – proximity POIs
- NTA/CDTA–Tract Equivalency (`hm78-6dwm`) – equity bridge
- ACS‑derived income at CDTA – income deciles for equity analysis

Methods highlights:
- Spatial joins and intersections with GeoPandas; distances/lengths in EPSG:2263; storage in EPSG:4326
- SoQL filters + pagination on Socrata; hourly aggregation from timestamps
- Normalization: rate = violations / lane‑miles by `NTA × hour`
- Proximity buffers: 100 m around schools/hospitals

## 3. Insights – Key findings and patterns
- **Equity (normalized):** Lower‑income districts exhibit higher violation rates per lane‑mile (large positive equity gradient between income decile 0 and 9).
- **Schools (100 m):** Near‑school shares spike at **7–9 AM** and **2–4 PM**, consistent with arrival/dismissal windows.
- **Hospitals (100 m):** Clear **AM ramp (7–8 AM)** and strongest **mid‑day/PM concentration (peak at 2 PM)**; no late‑night spike (10–11 PM).
- **Where/When playbook:** Ranked NTA × hour windows highlight a small set of places and times to prioritize staffing and curb/loading changes.

## 4. Narrative – Tell a story the audience can follow
We are three teammates—**Mohamed Hiba, Jean Carlo Chajon, and Shazadul Islam**—who set out to make ACE more actionable. Early on, we struggled to frame questions that were fair and operational: counts alone felt wrong, but what should be the denominator? After several false starts, we met with **@sudiptobt** (a prior Datathon winner). His feedback pushed us to normalize by exposure (lane‑miles) and to tie “when” to real‑world contexts (schools and hospitals).

From there, we:
1) Pulled Manhattan slices via SoQL, projected geometries to EPSG:2263, and computed **lane‑miles by NTA**.  
2) Joined ACE points to NTAs, extracted **hour of day**, and computed **rate = violations / lane‑miles** by `NTA × hour`.  
3) Buffered schools/hospitals by **100 m** to flag nearby violations and summarized **hourly shares** to detect time‑of‑day spikes.  
4) Mapped NTA→CDTA via the crosswalk and aggregated lane‑mile‑weighted rates to CDTA, then assigned **income deciles** for equity.  
5) Built a **Where/When action playbook** and an interactive **Folium + React** dashboard to communicate results.

The hardest parts were getting the data right (API quirks, geometry types, projections), choosing the correct normalization, and keeping the story tight. The mentor check‑in helped us simplify the problem to what matters operationally: “Where should we go, and at what hour?”

## 5. Action – Concrete steps or recommendations
- **Staff ACE during top NTA × hour windows** from the playbook; validate with field teams.
- **Near schools:** Focus operations and curb/loading management during **7–9 AM** and **2–4 PM**.
- **Near hospitals:** Prioritize **mid‑day/PM (peak ~2 PM)**; late‑night enforcement is lower‑yield.
- **Equity:** Use normalized rates to prioritize resources in lower‑income districts, not just where lanes are longest.
- **Next steps:** Add an ACE‑coverage denominator (ACE‑equipped lane‑miles), extend beyond Manhattan, and link to bus delay datasets to quantify benefits.
