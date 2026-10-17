# NFL Tendencies

A scouting and tendency dashboard for every NFL team, built on [nflfastR](https://www.nflfastr.com/) play-by-play.
It's the NFL counterpart to the [Dowling Catholic Football Analytics](https://github.com/zcalvert29/Dowling-Football)
app: the same tables, colors, takeaway cards, and tendency views, but with every number compared to the rest of the
league in the same situation.

---

## What's inside

A multipage **Streamlit** app. Pick a team (and, on a couple of pages, an opponent) in the sidebar.

| Area | What it does |
|---|---|
| **League Overview** | Every team's offense vs defense EPA and pass vs run EPA (team colors), plus a sortable league table with top/bottom-quarter shading |
| **Offense Overview** | EPA, success, explosive rate vs the NFL; results by down, personnel, formation, personnel + formation, score, and quarter, with takeaway cards and an EPA-vs-success chart |
| **Run Game** | EPA by run lane (ends, tackles, guards, middle) against the league in the same lane; runs by personnel, formation, box count, and distance; ball carriers |
| **Pass Game** | Pass-zone heatmap (direction × depth) against the league, by depth and formation, vs each coverage, man vs zone, clean vs pressured, by number of rushers; targets and passers |
| **Defense Overview / Pass / Run** | The same views from the defense's side: what it allowed, which packages it answers each personnel with, coverage mix by down & distance, blitz and pressure rates, box counts |
| **Red Zone** (offense and defense) | Trips, TD rate, points per trip vs the NFL; by down, personnel, and where the throws go |
| **Weekly Trends** | EPA, success, pass/rush EPA, and pass rate over expected game by game, with the league average |
| **Tendencies & Tells** | *Tells*: situations where a team's run/pass split is far from both its usual split and the league's, flagged only when two-proportion z-tests say it isn't noise. Plus a down & distance grid, a personnel → formation tendency tree, play sequencing, field position, 3rd downs, and 4th-down aggressiveness vs the league in the same spots |
| **Matchup** | One team's offense against another's defense, metric by metric with league ranks and who has the edge; personnel and coverage cross-checks |
| **Team Profiles** | League-percentile bars for offense and defense, two teams side by side |
| **Season Comparison** | One team, two seasons: offense and defense metrics, ranks, and the change |
| **Game Recap** | Win probability chart, head-to-head numbers, drives, and the biggest swings of any game |
| **Play Finder** | Filter by personnel, formation, coverage, field zone, or words in the play description; download as CSV |

### Filters
Season, season type, week range, down, distance, situation (neutral, garbage time removed, early downs, two-minute,
red zone, by half, leading/trailing), and a minimum-plays cutoff for tables. League comparisons always use the same
filters across all 32 teams, so every "vs NFL" number is apples to apples.

## Data

- **Play-by-play:** nflfastR via the [nflverse data releases](https://github.com/nflverse/nflverse-data/releases)
  (EPA, success, win probability, `xpass`/pass rate over expected, CPOE, pass direction and air yards, run gap).
- **Participation charting:** nflverse's FTN/NGS participation file adds offensive formation and personnel,
  defensive personnel, men in the box, number of pass rushers, pressure, and coverage (man/zone and shell).
  nflverse publishes it after each season, so **the season in progress has no personnel or coverage yet**; formation
  falls back to nflfastR's shotgun / under center flag, and those tables say so.
- **Teams:** names, colors, and logos from nflverse's `teams_colors_logos`.

The app downloads each season the first time it's needed and keeps a slimmed copy in `data/` (about 5 MB a season).
The current season is re-downloaded every 12 hours so new weeks show up.

## Project structure

```
app.py          Streamlit entrypoint: navigation, sidebar filters
views.py        Page bodies
data.py         Downloads nflfastR pbp + participation and builds the app's columns (personnel codes,
                distance/field buckets, pass zones, run lanes, sequencing)
tables.py       Crosstabs, table coloring, takeaway cards, EPA vs success scatter
charts.py       Pass zones, run lanes, weekly trends, league scatter, percentile bars, win probability
tendencies.py   Tells, down & distance grid, tendency tree, sequencing, field position, red zone, 3rd/4th downs
tests/          Unit tests for the data curation and tell detection
```

## Running locally

```bash
git clone https://github.com/zcalvert29/NFL-Tendencies.git
cd NFL-Tendencies
pip install -r requirements.txt
streamlit run app.py
```

To pre-download seasons (optional; the app does it on demand): `python data.py 2024 2025 2026`

Run the tests with:

```bash
python -m pytest tests/
```

## Definitions

- **EPA:** expected points added, from nflfastR's model. **Success:** a play with positive EPA (nflfastR's
  definition; the Dowling app uses the 40/70/100% yardage rule instead).
- **Explosive:** a run of 10+ or a pass of 20+ yards.
- **PROE:** pass rate over expected, from nflfastR's `xpass` model.
- **Personnel:** RBs (incl. FBs) then TEs, so "11" = 1 RB, 1 TE, 3 WR; "+OL" = an extra lineman.
- **Defensive package:** Base = 4 DBs, Nickel = 5, Dime = 6, Quarter = 7+.

## Tech stack
**Python** (Streamlit, pandas, NumPy, Altair, PyArrow) · nflverse / nflfastR data · pytest

## Author
**Zack Calvert** · [LinkedIn](https://linkedin.com/in/zack-calvert) · [Substack](https://zcalvert.substack.com)
