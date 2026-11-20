# Model — GTA VI trophy rarity forecast

## What the data does at launch

An earn rate is earners divided by players tracked. In launch week the
denominator explodes (Exophase tracked 638 players on launch night and 3,868
by the first refresh a few hours later, and keeps growing) while earners
accumulate only as players actually reach each trophy. So most rates FALL
sharply even though more people earn every trophy. Any forecaster that
extrapolates the rate itself is fitting the wrong curve.

## The split

The forecaster splits the rate into its two moving parts and fits each on
log time, `ln(1 + hours since launch)`:

- **Players tracked N(t):** log-linear fit, blended 75/25 with a
  square-root-time fit as a hedge against the launch surge lasting longer.
- **Earners E_i(t) = rate_i(t) x N(t), per trophy:** log-linear fit, with the
  slope shrunk toward the mean slope of the trophy's speed class. Speed
  class (fast / medium / slow) comes from the trophy text at run time: story
  trophies are reached quickly, grind and collectible trophies slowly.
- **Recombine:** forecast rate = E-hat / N-hat at the target time.
- A direct fit of the logit rate itself is blended in at 30% as a hedge.

Near-saturated trophies (rate at or above 90%) are forecast through the
non-earner count N - E instead, which is what actually moves there.

## Single-snapshot history

The practice histories are completely flat (one distinct read), and the real
final history may be nearly so if Exophase refreshes rarely. With a single
distinct state there is nothing to fit, so the forecaster falls back to a
dilution prior: N grows log-linearly through the two anchors the rules
document (638 players at launch, 3,868 at the first refresh), and earners
grow as (N_target / N_now) raised to a speed-class power — fast 0.85,
medium 0.45, slow 0.18. Those powers were picked on a backtest (see below),
deliberately between the middle and the best of the sensitivity runs rather
than at the argmin, to avoid overfitting one 55-trophy test.

## Spread

Each trophy's five quantiles come from a normal spread on the logit scale
around the median. The spread combines the fit residuals, the N-forecast
error, a random-walk term (0.5 x sqrt(days)), and the median refresh jump
seen in the history (floored 0.25, capped 0.8).

## Backtest

Out-of-sample test: history = the flat practice case at 2026-11-19T03:30Z,
target = the real Exophase snapshot fetched 2026-11-20T02:09Z (1,359 minutes
later). Pinball loss on the logit scale: house persistence baseline 2.2779,
house curve baseline 2.2779 (identical — the history is flat), this
forecaster 0.3617, with 90% interval coverage 1.00 and a typical miss of
0.069 percentage points. The data behind that test is a real public
Exophase page fetch, not a simulation.

## Limits

- No live Exophase denominator is known after the 2026-11-20 snapshot; the
  final history's player counts come from the house's own reads.
- The speed-class powers rest on a single 55-trophy backtest; the fitted
  path (used when the history has three or more distinct states) does not
  depend on them.
- The program is deterministic, standard library only, no network, no wall
  clock; it reads one case file from /input and writes `<id>.out` to /output.
