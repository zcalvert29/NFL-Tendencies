# NFL Tendencies

The [Dowling Catholic Football Analytics](https://github.com/zcalvert29/Dowling-Football) app's **Scout Opposing
Offense** page, rebuilt for any NFL offense on [nflfastR](https://www.nflfastr.com/) play-by-play. Same views,
colors, and tendency grading, with two additions throughout:

- **PROE (pass rate over expected):** how much more often a team throws than an average NFL team would in the exact
  same spots (down, distance, field position, score, time, timeouts), from nflfastR's `xpass` model.
- **League-wide situational averages:** every table can carry "NFL" columns, which are the league's numbers for the
  same row with the same filters (the NFL's pass rate on 3rd & long next to the team's 3rd & long). The run/pass
  bars mark the NFL's split, the run-lane and pass-zone pictures show and color against the NFL's number in the same
  lane or zone, and the red zone numbers show the difference from the NFL.

## Pages

### Tendencies

Pick an offense in the sidebar. Five tabs:

| Tab | What it shows |
|---|---|
| **Tendencies** | Down & distance grid (run/pass split with the NFL's split marked, PROE, most-used formation and personnel, tell / lean badges); PROE charts by down & distance and by game situation (their pass rate, expected pass rate, and the NFL's actual pass rate); offensive, formation, and personnel tendency tables; 3rd and 4th downs with NFL conversion rates; every 3rd/4th down call; best 10 plays |
| **Sequencing** | What they call after a run, a pass, a completion, an incompletion, a big gain, a loss, a play that worked or didn't, and on a new set of downs, each with the NFL's pass rate after the same kind of snap and PROE; by down after a run or a pass |
| **Red zone** | Trips, touchdown rate, scoring rate, points per trip, and trips per game against the NFL; how trips ended; how it's called in each zone inside the 20 with NFL pass rate, success, and EPA; formations and personnel inside the 20; run/pass by direction between the 11 and 20 and inside the 10 |
| **Run game** | Run-lane diagram (EPA and carries for each of nflfastR's seven lanes, colored against the NFL's EPA in the same lane); runs by lane, formation, distance, personnel, and box count; ball carriers; best runs |
| **Pass game** | Pass-zone heatmap (direction × depth, each zone shaded against the NFL's number in the same zone); pass chart of every throw; passes by depth, formation, coverage, and man/zone; targets; best passes |

**Sidebar filters:** season, season type, week, down, distance, situation (neutral, red zone, goal to go, backed
up, 3rd & short, short yardage, long yardage, two-minute), and a minimum-plays cutoff. NFL comparisons always use
the same filters across all 32 teams.

### Play-Calling Identity

Which offenses call it unlike an average NFL team in the same spots, and which are NFL average. Every snap is compared
with what the rest of the league does in that exact spot (regular season, score within 16, outside the last 2
minutes of a half), on three measures:

| Measure | Compared with |
|---|---|
| **Lean (PROE)** | nflfastR's `xpass`, re-centered on what the rest of the league actually did in the same spots that season (the model was fit on earlier seasons and expects more passing on 3rd/4th & 1-2 than today's NFL does) |
| **Shotgun over expected** | the rest of the league's shotgun rate in the same down × distance × field zone × score state |
| **3+ WR personnel over expected** | the rest of the league's 3+ WR rate in the same spots (seasons with FTN charting only) |

Each measure gets a game-clustered z-score (|z| ≥ 2 = detectable) and an empirical-Bayes shrunk value in units of
the real team-to-team spread. The **unorthodoxy index** is the root mean square of those: 0 = calls it like an
average team, 1 = one typical team-to-team spread away on each measure. *Unorthodox* = index ≥ 1.0; *NFL average* =
not detectably and materially (≥ 0.5 spreads) different on any measure; *Distinct* = in between. A measure a season
doesn't have (personnel in the season in progress) is left out of that season's index and table.

The table lists every value for exploration: pass, shotgun, 3+ WR, 2+ TE, and 2+ RB rates against expected, the
personnel mix distance, top personnel group, situational PROE against the NFL (early downs, 1st & 10, 2nd & long,
short yardage, red zone), and last season's rank. A reliability table shows how much of each measure is a real
team trait (split-half, Spearman-Brown): shotgun and personnel ≈ 0.95, lean 0.70-0.88, short-yardage PROE ≈ 0.6,
which is why the situational splits are kept out of the index.

| 2025 | 2026 (6 weeks, no personnel charting yet) |
|---|---|
| ![2025](docs/identity_2025.png) | ![2026](docs/identity_2026.png) |

More screenshots, including the full tables, are in [`docs/`](docs/).

## Data

- **Play-by-play:** nflfastR from the [nflverse data releases](https://github.com/nflverse/nflverse-data/releases):
  EPA, success, `xpass` / `pass_oe`, pass direction and air yards, run location and gap.
- **Participation charting:** nflverse's FTN participation file adds formation, personnel, men in the box, and
  coverage. nflverse publishes it after each season, so **the season in progress has no personnel, box count, or
  coverage yet**; formation falls back to nflfastR's shotgun / under center flag, and those tables say so.

The app downloads each season the first time it's picked and keeps a slimmed copy in `data/` (about 5 MB a
season). The season in progress is re-downloaded every 12 hours so new weeks show up.

## Project structure

```
app.py              Navigation between the pages
page_tendencies.py  Tendencies page: sidebar filters, tabs
page_identity.py    Play-Calling Identity page
identity.py         Play-calling identity measures (no Streamlit code)
data.py             Downloads nflfastR pbp + participation; builds the app's columns
common.py           Metrics, NFL comparison columns, table colors, tendency badges, split bars
tab_tendencies.py   Tendencies tab
tab_sequencing.py   Sequencing tab
tab_red_zone.py     Red zone tab
tab_run_game.py     Run game tab (run-lane diagram)
tab_pass_game.py    Pass game tab (pass zones)
pass_chart.py       The pass chart drawing
tests/              Unit tests
docs/               Screenshots
```

## Running locally

```bash
git clone https://github.com/zcalvert29/NFL-Tendencies.git
cd NFL-Tendencies
pip install -r requirements.txt
streamlit run app.py
```

Optional: `python data.py 2024 2025 2026` downloads seasons ahead of time. Run the tests with `python -m pytest`.

## Definitions

- **EPA:** expected points added (nflfastR). **Success:** a play with positive EPA (nflfastR's definition; the
  Dowling app uses the 40/70/100% yardage rule).
- **Explosive:** a run of 10+ or a pass of 20+ yards.
- **Personnel:** RBs (incl. FBs) then TEs, so "11" = 1 RB, 1 TE, 3 WR; "+OL" = an extra lineman.
- **Tell / lean / small sample:** how much to trust a 70%+ split, graded by the Wilson lower bound (90%), as in the
  Dowling app.
- **Colors:** from the scouted offense's side. With an NFL column, green/red = clearly better/worse than the NFL in
  that row. Without one, the Dowling offense rule. Rows under 5 plays aren't colored; rows under 10 are grayed.

## Author
**Zack Calvert** · [LinkedIn](https://linkedin.com/in/zack-calvert) · [Substack](https://zcalvert.substack.com)
