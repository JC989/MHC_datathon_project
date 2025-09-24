import "./app.css"

function App() {
  return (
    <div className="app">
      <header className="header">
        <h1>ACE Violations Analysis Dashboard</h1>
        <p className="subtitle">The following visualizations provide insights into ACE violations in Manhattan.</p>
      </header>

      <main className="main-content">
        {/* Graph Section 1 */}
        <section className="graph-section">
          <div className="graph-container">
            <img src="./src/photos/ACE_Violations_Hospital.png" width="600" alt="ACE Violations by Hospital" />
          </div>
          <div className="explanation">
            <h3>ACE Violations by Hospital</h3>
            <p>
              This chart shows the distribution of ACE violations across different hospitals in Manhattan, highlighting
              areas with higher rates of violations.
            </p>
          </div>
        </section>

        {/* Graph Section 2 */}
        <section className="graph-section">
          <div className="graph-container">
            <img
              src="./src/photos/proximity_hospitals_by_hour_mn.png"
              width="600"
              alt="Proximity to Hospitals by Hour"
            />
          </div>
          <div className="explanation">
            <h3>Proximity to Hospitals by Hour</h3>
            <p>
              This chart shows how proximity to hospitals varies by hour, which may correlate with the timing of ACE
              violations.
            </p>
          </div>
        </section>

        {/* Graph Section 3 */}
        <section className="graph-section">
          <div className="graph-container">
            <img
              src="./src/photos/proximity_schools_hourly_share_mn.png"
              width="600"
              alt="Proximity to Schools and ACE Violations"
            />
          </div>
          <div className="explanation">
            <h3>Proximity to Schools and ACE Violations</h3>
            <p>
              This chart illustrates the relationship between the proximity to schools and the rate of ACE violations,
              indicating potential areas of concern.
            </p>
          </div>
        </section>

        {/* Graph Section 4 */}
        <section className="graph-section">
          <div className="graph-container">
            <img src="./src/photos/proximity_schools_by_hour_mn.png" width="600" alt="Proximity to Schools by Hour" />
          </div>
          <div className="explanation">
            <h3>Proximity to Schools by Hour</h3>
            <p>
              This chart shows how proximity to schools varies by hour, which may correlate with the timing of ACE
              violations.
            </p>
          </div>
        </section>

        {/* Graph Section 5 */}
        <section className="graph-section">
          <div className="graph-container">
            <img
              src="./src/photos/equity_rate_vs_income_decile_mn.png"
              width="600"
              alt="Equity Rate vs Income Decile"
            />
          </div>
          <div className="explanation">
            <h3>Equity Rate vs Income Decile</h3>
            <p>
              This chart compares the equity rate across different income deciles, shedding light on disparities in ACE
              violations.
            </p>
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
