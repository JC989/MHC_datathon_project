# ACE Bus Lane Violations Analysis

## Project Overview

This datathon project analyzes MTA's Automated Camera Enforcement (ACE) bus lane violations with a focus on **equity, schools, and hospitals**. We normalize violation rates by bus lane miles to ensure fair comparisons across neighborhoods and identify actionable insights for the MTA.

## Key Research Questions

1. **Equity (normalized):** Are violation rates per bus-lane mile higher in lower-income neighborhoods?
2. **Schools:** Within 100m of schools, do rates spike at 7–9 AM & 2–4 PM vs other hours?
3. **Hospitals:** Within 100m of hospitals, do rates spike at 6–8 AM, 2–4 PM, 10–11 PM?

## What We're Doing

- Finding places where cars block bus lanes the most, especially **near schools and hospitals**
- Checking **when** violations spike (morning drop-off, afternoon pickup, hospital shift changes)
- Making it **fair** by comparing **rates** (violations per mile of bus lane) instead of raw counts
- Checking if problem spots are **more common in lower-income neighborhoods**
- Suggesting **practical curb fixes** (loading zones, physical protection, targeted enforcement)

## What We'll Show

1. A **map** of neighborhoods colored by **violation rate** (not counts), with bus lanes drawn
2. A simple **hour-by-hour chart** showing spikes near **schools and hospitals**
3. A **fairness chart** showing how rates change from **lower-income to higher-income** areas

## Project Structure

```
datathon/
├── notebooks/                    # Jupyter notebooks for analysis
│   ├── 01_ace_violations_data_fetch.ipynb
│   ├── 02_ace_violations_processing.ipynb
│   └── 06_ace_violations_visualization.ipynb
├── src/                         # Python source code
│   └── ace_violations_analysis.py
├── data/                        # Data storage
│   ├── raw/                     # Raw data from APIs
│   └── processed/               # Processed data and results
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the analysis:**
   ```bash
   python src/ace_violations_analysis.py
   ```

3. **Or use Jupyter notebooks:**
   ```bash
   jupyter lab
   ```

## Datasets Used

- **ACE Violations (kh8p-hcbm)** - Main violation data with timestamps and locations
- **Bus Lanes (rx8t-6euq)** - For lane miles calculation (exposure denominator)
- **NTA Boundaries (9nt8-h7nd)** - For neighborhood analysis
- **Schools (s3k6-pzi2)** - For proximity analysis
- **Hospitals (f7b6-v6v3)** - For proximity analysis
- **Income Data (5uac-w243)** - For equity analysis

## Key Metrics

- **Violation rate (main KPI):** `rate = violations / lane_miles` by NTA × hour_of_day
- **POI proximity flags:** within 100m of schools and hospitals
- **Equity gradient:** rate vs income decile by NTA

## Expected Insights

- Violation rates normalized by bus lane miles (fair comparison)
- Time patterns around schools (7-9 AM, 2-4 PM) and hospitals (6-8 AM, 2-4 PM, 10-11 PM)
- Higher violation rates in lower-income neighborhoods
- Actionable recommendations for MTA enforcement and infrastructure

## What This Gives the MTA

- Exactly **where** and **when** to act (the blocks and time windows)
- Low-cost, concrete changes that make buses faster and more reliable for riders
- Evidence-based approach to resource allocation
- Fair comparison across neighborhoods using normalized rates

## Authors

Datathon Team - 2025