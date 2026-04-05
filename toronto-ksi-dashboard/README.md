# Toronto KSI Collision Hotspot Dashboard

An interactive Tableau dashboard analyzing Killed or Seriously Injured (KSI) collision events across Toronto from 2006 to 2023, built for city executives and public safety planners.

## Decision Question

Where and when do the most dangerous KSI collisions occur, and which contributing factors are associated with them?

## Data Source

Toronto KSI Collision Records (2006--2023) via the City of Toronto Open Data Portal -- 18,957 records spanning 6,872 unique KSI events. Key fields include intersection name, geographic coordinates, road classification, injury type, involved party (pedestrian, cyclist, motor vehicle), and four binary contributing-factor flags: Aggressive Driving, Alcohol Involved, Red Light Infraction, and Speeding.

## Dashboard Components

| Component | Purpose |
|-----------|---------|
| BAN KPI tiles | Total KSI count and contributing-factor shares, updated dynamically with every filter |
| Filter panel | Year range slider, Time of Day, and four contributing-factor toggles for cross-filtering |
| Geographic hotspot map | Proportional symbol map of the 20 most dangerous intersections by KSI count |
| Top 20 ranked list | Horizontal bar chart of intersections sorted by cumulative KSI count |
| Time of Day mix | 100% stacked bar decomposing collisions across five time-of-day periods |

## Key Findings

- **Aggressive driving** is flagged in 50.7% of KSI collisions at the Top 20 hotspot intersections -- more than all other contributing factors combined
- **Night hours** (6 PM--midnight) carry the largest risk share at 29.85% of KSI events
- **Red-light infractions** appear in ~7.5% of events, supporting the case for Automated Speed Enforcement and red-light camera expansion
- **Finch Ave W & Weston Rd** consistently ranks as the single most dangerous hotspot across multiple filter configurations

## Files

| File | Description |
|------|-------------|
| `Toronto_KSI_Dashboard.twbx` | Packaged Tableau workbook (includes embedded data) |
| `Toronto_KSI_Collision_Hotspot_Report.pdf` | Full project report with design rationale and findings |

## Tools

Tableau Desktop, City of Toronto Open Data Portal
