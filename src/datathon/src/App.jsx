import "./App.css"
import { useEffect, useState } from "react"

function App() {
  const [meanRate, setMeanRate] = useState("—")
  const [peakHospHour, setPeakHospHour] = useState("—")
  const [equityGradient, setEquityGradient] = useState("—")
  const [playbookRows, setPlaybookRows] = useState([])

  useEffect(() => {
    function parseCSV(text) {
      const lines = text.trim().split(/\r?\n/)
      if (lines.length < 2) return []
      const headers = lines[0].split(",")
      return lines.slice(1).filter(Boolean).map(line => {
        const cols = line.split(",")
        const obj = {}
        headers.forEach((h, i) => { obj[h] = cols[i] })
        return obj
      })
    }

    // KPI 1: Citywide mean rate from rates_by_nta_hour_manhattan.csv
    fetch("/data/processed/rates_by_nta_hour_manhattan.csv")
      .then(r => r.text())
      .then(t => {
        const rows = parseCSV(t)
        const rates = rows.map(r => parseFloat(r.rate)).filter(v => Number.isFinite(v))
        if (rates.length > 0) {
          const avg = rates.reduce((a, b) => a + b, 0) / rates.length
          setMeanRate(avg.toFixed(2))
        }
      }).catch(() => {})

    // KPI 2: Peak near-hospital hour from proximity_hospitals_hourly_share_mn.csv
    fetch("/data/processed/proximity_hospitals_hourly_share_mn.csv")
      .then(r => r.text())
      .then(t => {
        const rows = parseCSV(t)
        let best = null
        rows.forEach(r => {
          const share = parseFloat(r.share_near_hospital)
          const hour = parseInt(r.hour, 10)
          if (Number.isFinite(share) && Number.isFinite(hour)) {
            if (!best || share > best.share) best = { hour, share }
          }
        })
        if (best) setPeakHospHour(`${best.hour}:00`)
      }).catch(() => {})

    // KPI 3: Equity gradient (bottom vs top income decile)
    fetch("/data/processed/equity_rates_by_income_decile_mn.csv")
      .then(r => r.text())
      .then(t => {
        const rows = parseCSV(t)
        const byDecile = Object.fromEntries(rows.map(r => [String(r.income_decile), parseFloat(r.mean_rate)]))
        const bottom = byDecile["0"]
        const top = byDecile["9"]
        if (Number.isFinite(bottom) && Number.isFinite(top) && top !== 0) {
          const pct = ((bottom - top) / Math.abs(top)) * 100
          const sign = pct >= 0 ? "+" : ""
          setEquityGradient(`${sign}${pct.toFixed(0)}%`)
        }
      }).catch(() => {})

    // Where/When playbook (top 10 rows)
    fetch("/data/processed/where_when_action_playbook_mn.csv")
      .then(r => r.text())
      .then(t => {
        const rows = parseCSV(t)
        setPlaybookRows(rows.slice(0, 10))
      }).catch(() => {})
  }, [])

  return (
    <div className="app">
      <header className="header">
        <h1>ACE Violations Analysis Dashboard</h1>
        <p className="subtitle">After normalizing for exposure (rate = violations / bus-lane miles), we show where and when ACE violations spike in Manhattan and how rates vary by income.</p>
      </header>

      <main className="main-content">
        {/* KPI chips */}
        <section className="kpi-row">
          <div className="kpi-chip">
            <div className="kpi-title">Mean rate (violations / lane‑mile)</div>
            <div className="kpi-value">{meanRate}</div>
          </div>
          <div className="kpi-chip">
            <div className="kpi-title">Peak near‑hospital hour</div>
            <div className="kpi-value">{peakHospHour}</div>
          </div>
          <div className="kpi-chip">
            <div className="kpi-title">Equity gradient (Decile 0 vs 9)</div>
            <div className="kpi-value">{equityGradient}</div>
          </div>
        </section>
        {/* Research Questions */}
        <section className="graph-section">
          <div className="explanation">
            <h3>Research Questions</h3>
            <ul>
              <li><b>RQ1 – Equity (normalized):</b> Are violation rates per bus‑lane mile higher in lower‑income neighborhoods?</li>
              <li><b>RQ2 – Schools:</b> Within 100 m of schools, do rates spike at 7–9 AM and 2–4 PM vs other hours?</li>
              <li><b>RQ3 – Hospitals:</b> Within 100 m of hospitals, do rates spike at 6–8 AM, 2–4 PM, 10–11 PM?</li>
            </ul>
            <p style={{marginTop: "8px"}}><i>We report rate = violations / lane‑miles. Counts alone can be misleading.</i></p>
          </div>
        </section>

        {/* RQ1 – Equity (normalized) */}
        <section className="graph-section">
          <div className="graph-container">
            <img
              src="/img/equity_rate_vs_income_decile_mn.png"
              width="600"
              alt="Equity Rate vs Income Decile"
            />
          </div>
          <div className="explanation">
            <h3>RQ1 – Equity (normalized)</h3>
            <p><b>What we test:</b> rate = violations / lane_miles, rolled up to CDTA and grouped into income deciles.</p>
            <p><b>How we computed:</b> rates_by_nta_hour_manhattan.csv → NTA→CDTA (hm78‑6dwm) → lane‑mile–weighted CDTA rates → deciles from CDTA median income (ACS/CCC).</p>
            <p><b>Finding:</b> Normalized violation rates are higher in lower‑income districts (per lane‑mile), not just where there’s more bus lane.</p>
            <p style={{marginTop: "6px"}}>
              CSV: <a href="https://github.com/JC989/MHC_datathon_project/blob/main/data/processed/equity_rates_by_income_decile_mn.csv" target="_blank" rel="noreferrer">equity_rates_by_income_decile_mn.csv</a>
            </p>
          </div>
        </section>

        {/* RQ2 – Schools (100 m) */}
        <section className="graph-section">
          <div className="graph-container">
            <img
              src="/img/proximity_schools_hourly_share_mn.png"
              width="600"
              alt="Share of violations near schools by hour"
            />
          </div>
          <div className="explanation">
            <h3>RQ2 – Schools (100 m)</h3>
            <p><b>What we test:</b> Is the mix of violations near schools higher during 7–9 and 14–16?</p>
            <p><b>How we computed:</b> Buffer schools 100 m (EPSG:2263), flag ACE points within buffer, compute hourly shares: near_school / total that hour.</p>
            <p><b>Finding:</b> Near‑school shares spike at 7–9 AM and 2–4 PM → target post‑stop loading and enforcement windows.</p>
            <p style={{marginTop: "6px"}}>
              CSV: <a href="https://github.com/JC989/MHC_datathon_project/blob/main/data/processed/proximity_schools_hourly_share_mn.csv" target="_blank" rel="noreferrer">proximity_schools_hourly_share_mn.csv</a>
            </p>
          </div>
        </section>

        {/* RQ3 – Hospitals (100 m) */}
        <section className="graph-section">
          <div className="graph-container">
            <img
              src="/img/proximity_hospitals_hourly_share_mn.png"
              width="600"
              alt="Share of violations near hospitals by hour"
            />
          </div>
          <div className="explanation">
            <h3>RQ3 – Hospitals (100 m)</h3>
            <p><b>What we test:</b> Is the mix of violations near hospitals higher during 6–8, 14–16, 22–23?</p>
            <p><b>How we computed:</b> Buffer hospitals 100 m, flag ACE points, hourly shares near_hospital / total.</p>
            <p><b>Finding:</b> Clear AM ramp (7–8 AM) and strongest mid‑day/PM concentration (peak at 2 PM); no late‑night spike.</p>
            <p style={{marginTop: "6px"}}>
              CSV: <a href="https://github.com/JC989/MHC_datathon_project/blob/main/data/processed/proximity_hospitals_hourly_share_mn.csv" target="_blank" rel="noreferrer">proximity_hospitals_hourly_share_mn.csv</a>
            </p>
          </div>
        </section>

        {/* Removed redundant sections */}

        {/* Map & Where/When */}
        <section className="graph-section fullwidth">
          <div className="graph-container">
            <div style={{border:"1px solid #ddd", padding:"12px", borderRadius:"6px"}}>
              <h3 style={{marginTop:0}}>Interactive Map</h3>
              <p style={{marginBottom:"8px"}}>Choropleth of NTA mean rate + bus lanes + schools/hospitals + “Action: Top 3” pins.</p>
              {/* Embed from public assets; full viewport height */}
              <div style={{width:"100%", height:"80vh", margin:"0 auto"}}>
                <iframe
                  title="Manhattan Map"
                  src="/reports/dashboards/map_manhattan_quickcheck.html"
                  style={{border:"0", width:"100%", height:"100%"}}
                />
              </div>
              <div style={{marginTop:"8px"}}>
                <a className="button" href="https://github.com/JC989/MHC_datathon_project/blob/main/reports/dashboards/map_manhattan_quickcheck.html" target="_blank" rel="noreferrer">Open Map on GitHub</a>
              </div>
            </div>
          </div>
          <div className="explanation">
            <h3>Where/When to Act</h3>
            <p>Top windows by normalized rate (violations per lane‑mile):</p>
            <ul>
              <li><a href="https://github.com/JC989/MHC_datathon_project/blob/main/data/processed/where_when_action_playbook_mn.csv" target="_blank" rel="noreferrer">Where/When Action Playbook (top 12 with recommendations)</a></li>
              <li><a href="https://github.com/JC989/MHC_datathon_project/blob/main/data/processed/where_when_top_overall_mn.csv" target="_blank" rel="noreferrer">Top NTA×hour by rate</a></li>
              <li><a href="https://github.com/JC989/MHC_datathon_project/blob/main/data/processed/where_when_top_per_hour_mn.csv" target="_blank" rel="noreferrer">Top 5 per hour</a></li>
            </ul>
            {playbookRows.length > 0 && (
              <div style={{marginTop: "12px", overflowX: "auto"}}>
                <table className="simple-table">
                  <thead>
                    <tr>
                      <th>NTA</th>
                      <th>Hour</th>
                      <th>Rate</th>
                      <th>Window</th>
                      <th>Recommendation</th>
                    </tr>
                  </thead>
                  <tbody>
                    {playbookRows.map((r, idx) => (
                      <tr key={idx}>
                        <td>{r.ntaname || r.NTA || r.nta2020 || ""}</td>
                        <td>{r.hour}</td>
                        <td>{r.rate ? Number(r.rate).toFixed(2) : ""}</td>
                        <td>{r.window || ""}</td>
                        <td>{r.recommendation || ""}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </section>

        {/* Downloads / Links */}
        <section className="graph-section">
          <div className="explanation">
            <h3>Downloads</h3>
            <ul>
              <li><a href="https://github.com/JC989/MHC_datathon_project/blob/main/data/processed/where_when_action_playbook_mn.csv" target="_blank" rel="noreferrer">Where/When Action Playbook (CSV)</a></li>
              <li><a href="https://github.com/JC989/MHC_datathon_project/blob/main/data/processed/where_when_top_overall_mn.csv" target="_blank" rel="noreferrer">Top NTA×hour by rate (CSV)</a></li>
              <li><a href="https://github.com/JC989/MHC_datathon_project/blob/main/data/processed/where_when_top_per_hour_mn.csv" target="_blank" rel="noreferrer">Top 5 per hour (CSV)</a></li>
            </ul>
          </div>
        </section>

        {/* Supplemental / Definitions */}
        <section className="graph-section">
          <div className="explanation">
            <h3>Supplemental / Definitions</h3>
            <ul>
              <li><b>Rate</b>: violations per lane‑mile (computed by NTA × hour).</li>
              <li><b>Lane‑miles</b>: total bus‑lane length within an NTA, measured in miles.</li>
              <li><b>Share near schools/hospitals</b>: near_POI / total violations that hour (Manhattan).</li>
              <li><b>Equity gradient</b>: ((rate in income decile 0 − decile 9) / decile 9) × 100.</li>
              <li><b>NTA/CDTA</b>: Neighborhood Tabulation Area / Community District Tabulation Area.</li>
              <li><b>ACE</b>: Automated Camera Enforcement violations captured on bus corridors.</li>
              <li><b>EPSG</b>: 4326 (WGS84 lat/lon) for storage; 2263 (NY Long Island ftUS) for accurate buffers/lengths.</li>
              <li><b>Buffer 100 m</b>: computed in EPSG:2263; 100 m ≈ 328.1 ft.</li>
              <li><b>SoQL & pagination</b>: API filters via $where; large pulls use $limit/$offset.</li>
              <li><b>Normalization note</b>: rates normalize by lane‑miles; proximity “shares” are mix metrics (not exposure‑normalized).</li>
            </ul>
          </div>
        </section>

      </main>

      <footer className="footer">
        <p>© 2025 MHC++ Datathon Project</p>
      </footer>
    </div>
  )
}

export default App
