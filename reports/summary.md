# ACE Bus Lane Violations Analysis - Project Summary

## Executive Summary

This datathon project successfully analyzed MTA's Automated Camera Enforcement (ACE) bus lane violations with a focus on **equity, schools, and hospitals**. The analysis normalizes violation rates by bus lane miles to ensure fair comparisons across neighborhoods and provides actionable insights for the MTA.

## Key Achievements

✅ **Complete Analysis Pipeline**: Built a comprehensive data analysis system that fetches, processes, and analyzes ACE violations data

✅ **Fair Rate Calculation**: Implemented violation rates normalized by bus lane miles (violations per lane mile) instead of raw counts

✅ **Equity Analysis**: Analyzed violation patterns by neighborhood income levels to identify equity issues

✅ **Time Pattern Analysis**: Identified peak violation times around schools and hospitals

✅ **Actionable Insights**: Generated specific recommendations for MTA enforcement and infrastructure improvements

## Research Questions Addressed

### 1. Equity Analysis (Normalized)
**Question**: Are violation rates per bus-lane mile higher in lower-income neighborhoods?

**Answer**: Yes, the analysis shows that violation rates are significantly higher in lower-income neighborhoods, indicating an equity issue that needs to be addressed.

### 2. School Proximity Analysis
**Question**: Within 100m of schools, do rates spike at 7–9 AM & 2–4 PM vs other hours?

**Answer**: Yes, areas near schools show elevated violation rates during school drop-off (7-9 AM) and pickup (2-4 PM) times, with clear patterns indicating parent/guardian behavior.

### 3. Hospital Proximity Analysis
**Question**: Within 100m of hospitals, do rates spike at 6–8 AM, 2–4 PM, 10–11 PM?

**Answer**: Hospital areas show consistent violation patterns throughout the day, with slight peaks during shift changes, indicating different behavioral patterns compared to schools.

## Key Findings

1. **Equity Issue**: Violation rates are significantly higher in lower-income neighborhoods
2. **Time Patterns**: Clear spikes during school drop-off (7-9 AM) and pickup (2-4 PM) times
3. **Geographic Clustering**: Certain neighborhoods show consistently high violation rates regardless of time
4. **Institution-Specific Patterns**: Schools show time-based spikes while hospitals show more consistent patterns

## Technical Implementation

### Data Sources
- **ACE Violations**: NYC Open Data (kh8p-hcbm) - 3.78M+ records
- **Bus Lanes**: NYC Open Data (rx8t-6euq) - 3,732 records  
- **Neighborhoods**: NYC Open Data (9nt8-h7nd) - NTA boundaries
- **Schools**: NYC Open Data (s3k6-pzi2) - School locations
- **Hospitals**: NYC Open Data (f7b6-v6v3) - Hospital locations
- **Income Data**: NYC Open Data (5uac-w243) - ACS income data

### Methodology
- **Normalization**: Violation rates calculated as violations per bus lane mile
- **Proximity Analysis**: Violations within 100 meters flagged for analysis
- **Equity Analysis**: Neighborhoods grouped into income deciles
- **Time Analysis**: Hourly patterns identified and analyzed

### Key Metrics
- **Violation rate (main KPI)**: `rate = violations / lane_miles` by NTA × hour_of_day
- **POI proximity flags**: within 100m of schools and hospitals
- **Equity gradient**: rate vs income decile by NTA

## Recommendations for MTA

### 1. Targeted Enforcement
- Focus resources on high-violation areas during peak hours
- Prioritize enforcement in lower-income neighborhoods with high violation rates

### 2. Infrastructure Improvements
- Install physical barriers or enhanced signage in problem areas
- Create short-term loading zones near schools during drop-off/pickup times

### 3. Dynamic Solutions
- Consider time-based enforcement strategies around schools and hospitals
- Implement dynamic pricing for parking near schools and hospitals during peak times

### 4. Equity Focus
- Ensure enforcement resources are distributed fairly across neighborhoods
- Address root causes of higher violation rates in lower-income areas

## Project Deliverables

### 1. Analysis Code
- **`src/ace_violations_analysis.py`**: Complete analysis pipeline
- **Jupyter Notebooks**: Step-by-step analysis documentation
- **Requirements**: All necessary dependencies

### 2. Visualizations
- **Hourly Patterns**: Charts showing violation spikes by time
- **Equity Analysis**: Income vs violation rate relationships
- **Proximity Analysis**: School and hospital proximity patterns

### 3. Data
- **Raw Data**: Fetched from NYC Open Data APIs
- **Processed Data**: Cleaned and analyzed datasets
- **Results**: Violation rates and insights

### 4. Documentation
- **README.md**: Project overview and setup instructions
- **index.html**: Interactive dashboard
- **Summary**: This comprehensive project summary

## Impact and Value

This analysis provides the MTA with:

1. **Evidence-Based Decision Making**: Data-driven insights for resource allocation
2. **Fair Comparison**: Normalized rates ensure equitable analysis across neighborhoods
3. **Actionable Recommendations**: Specific, implementable solutions
4. **Equity Focus**: Addresses social justice concerns in transportation enforcement
5. **Cost-Effective Solutions**: Low-cost interventions with high impact potential

## Next Steps

1. **Deploy Analysis**: Run with real-time data feeds
2. **Stakeholder Engagement**: Present findings to MTA leadership
3. **Pilot Programs**: Implement recommended solutions in high-priority areas
4. **Monitoring**: Track effectiveness of implemented solutions
5. **Expansion**: Apply methodology to other transportation enforcement areas

## Conclusion

This project successfully demonstrates how data science can be used to address real-world transportation challenges while promoting equity and fairness. The analysis provides the MTA with concrete, actionable insights that can improve bus service reliability and address social justice concerns in transportation enforcement.

The methodology developed here can be replicated for other cities and transportation systems, making it a valuable contribution to the field of transportation data science.

---

**Project Team**: Datathon 2025  
**Date**: September 2025  
**Status**: Complete
