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
with what the rest of the league does in that exact spot (the team is left out of its own baseline), on four
measures:

| Measure | Compared with |
|---|---|
| **Lean (PROE)** | nflfastR's `xpass`, re-centered on what the rest of the league actually did in the same spots that season (the model was fit on earlier seasons and expects more passing on 3rd/4th & 1-2 than today's NFL does) |
| **Shotgun over expected** | the rest of the league's shotgun rate in the same down × distance × field zone × score state |
| **3+ WR personnel over expected** | the rest of the league's 3+ WR rate in the same spots (seasons with FTN charting only) |
| **Situational play calling** | in 13 situations (1st & 10, 2nd & short/long, 3rd/4th & 1-2, 3rd & medium, red zone, goal to go, backed up, leading, trailing, after a run, after a pass, first 15 snaps): how much more or less the team throws than the rest of the league does there, *beyond its own overall lean* |

Each measure gets a game-clustered z-score and an empirical-Bayes shrunk value in units of the real team-to-team
spread (noisy situations shrink toward 0, so they barely move the index; each situation needs a Bonferroni-adjusted
z to be flagged). The **unorthodoxy index** is the root mean square of those: 0 = calls it like an average team,
1 = one typical team-to-team spread away on each measure. *Unorthodox* = index ≥ 1.0; *NFL average* = not detectably
and materially different on any measure; *Distinct* = in between.

**Reading the index:** the unit is a typical team-to-team gap, so 1.0 = about one typical gap from average across the
measures. It has no direction (pass-heavy and run-heavy both count) and no grade (unusual isn't good or bad). Even the
most average offenses land around 0.3-0.4, since squares can't cancel and some noise survives the shrinkage; across
2021-25 the median team is about 0.77, the top quarter 1.07+, and the top 10% 1.3+.

**On the page:** the index as bars in each team's primary color with its logo on top; offensive EPA per play vs the
index; lean vs shotgun, lean vs personnel, and lean vs situational play calling (logos only); pass rate over expected
per team as bars in team colors with logos; a table of every value; and split-half reliabilities. Every chart uses team colors and logos. Logos are downloaded once from ESPN's CDN,
shrunk, cached in `data/logos/`, and embedded in the charts (a hotlinked image can fail to draw on Vega's canvas).

**Sidebar filters:** season (defaults to the latest), week range, score window, last 2 minutes of a half, which
measures and situations go into the index, and which teams to show (conference, division, profile, highlight).
A measure a season doesn't have (personnel in the season in progress) is left out of that season's index and table.

| 2026 (default view) | 2025 |
|---|---|
| ![2026](docs/identity_2026.png) | ![2025](docs/identity_2025.png) |

The logos in these two screenshots are stand-in badges (team color and abbreviation): the machine that took them
couldn't reach ESPN's CDN. The app shows the real logos.

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
