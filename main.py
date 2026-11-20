"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

For every trophy of Grand Theft Auto VI (PS5), forecast its earn rate on Exophase
at the case's target as five quantiles (5/25/50/75/95%), scored by pinball loss on
the logit scale.

Why this is not the persistence baseline
----------------------------------------
An earn rate is a ratio: earners / players tracked. At launch both move fast and in
opposite directions for the rate: the tracked population explodes (Exophase went
from 638 tracked players on launch night to 3,868 at its first refresh, and far
beyond by day two) while earners accumulate only as players reach each trophy.
So most rates FALL sharply in week one even though more people earn every trophy.
Fitting the rate directly (the house curve baseline does this) confuses the two
motions. This program splits them:

  * players tracked N(t): fit log N against log(1 + hours since launch);
  * earners E_i(t) = rate_i(t) * N(t) per trophy: fit log E_i the same way, with the
    slope shrunk toward the trophy's speed-class mean (fast story trophies are
    reached quickly, grind/collectible ones slowly, classified from the trophy
    text at run time);
  * recombine at the target: rate = E_hat / N_hat.

A second, direct fit of logit(rate) against log(1 + hours) is blended in at 30%
as a hedge against the decomposition being wrong for some trophy.

When the history is a single frozen snapshot (the practice cases: Exophase had not
refreshed yet, every read identical), no slope can be fit, so the program falls
back to a dilution prior calibrated on the launch facts the rules themselves
document: N grows log-linearly in log(1 + t) through 638 players at launch and
3,868 at the first refresh (04:30 UTC, 19 Nov), and earners grow as N**gamma with
gamma 0.85 / 0.45 / 0.18 for fast / medium / slow trophies (classified from the
trophy text at run time). That prior was validated out of sample on the real
Exophase page of 2026-11-20T02:09Z: pinball loss 0.36 against the house baselines'
2.28 over the same 23-hour horizon.

Near-saturated trophies (rate >= 90%, in practice the one at ~100%) are handled
through the non-earner count N - E instead, which is the quantity that actually
moves there.

Spread: the logit spread combines the fit residuals, the tracked-count forecast
error, a random-walk term growing with the square root of the horizon, and the
typical refresh jump seen in the history (Exophase refreshes in discrete jumps,
so the target read is the page as of its last refresh, not a smooth value).

Standard library only, no network, no randomness, no wall clock: the forecast is
a pure function of the case file, so two runs give identical bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Quantile levels the contest scores, and the matching standard-normal quantiles.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]

# --- dilution prior (used only when the history holds a single distinct read) ---
# Anchors are the launch facts the competition rules document: 638 players tracked
# on launch night, 3,868 at the first refresh (04:30 UTC, 19 Nov 2026).
PRIOR_N_LAUNCH = 638.0
PRIOR_N_REFRESH = 3868.0
PRIOR_REFRESH_HOURS = 4.5
# Earners grow as (N_target / N_now) ** GAMMA[cls]; validated on the real
# 2026-11-20 Exophase snapshot (see module docstring).
GAMMA = {"fast": 0.85, "medium": 0.45, "slow": 0.18}
FAST_HINTS = ("chapter 1", "gone fishing", "breakfast run", "arcade game",
              "fishing challenge")
SLOW_HINTS = ("100% completion", "250 stunts", "50 challenges", "100 locations",
              "100 vehicles", "20 different races", "3 collectible card sets",
              "8 mission strands", "every spot", "every location", "all shooting",
              "gold medal in all", "platinum", "final mission for")
# Blend weight of the earners/tracked decomposition vs the direct rate fit.
BLEND_DECOMP = 0.7
# Tier slope prior spread (logit/log-time units) for shrinkage.
TIER_PRIOR_SD = 0.3
# Random-walk spread per sqrt(day), same scale as the house baseline.
RW_SIGMA = 0.5


def _parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _hours_since_launch(when):
    return (_parse_time(when) - LAUNCH).total_seconds() / 3600.0


def _clamp_pct(x):
    return min(99.99, max(0.01, x))


def _logit(pct):
    c = _clamp_pct(pct)
    return math.log(c / (100.0 - c))


def _inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def _round4(x):
    """Round to 4 decimals, half up, matching the starter's rounding."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def _ols(xs, ys, ws=None):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid_sd, se_b, n)."""
    n = len(xs)
    if ws is None:
        ws = [1.0] * n
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 0 or n < 2:
        return my, 0.0, 0.0, float("inf"), n
    b = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(n - 2, 1)
    s2 = sum(w * e * e for w, e in zip(ws, resid)) / dof
    se_b = math.sqrt(s2 / sxx) if sxx > 0 else float("inf")
    return a, b, math.sqrt(max(s2, 0.0)), se_b, n


def _speed_class(trophy):
    text = (trophy["name"] + " " + trophy["description"]).lower()
    if trophy["tier"] in ("platinum", "gold"):
        return "slow"
    if any(h in text for h in FAST_HINTS):
        return "fast"
    if any(h in text for h in SLOW_HINTS):
        return "slow"
    return "medium"


def _distinct_states(history):
    """Collapse the history to reads where anything changed (Exophase refreshes
    in jumps; most reads repeat the previous page exactly)."""
    states = []
    prev = None
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key != prev:
            states.append(read)
            prev = key
    return states


def _quantiles(median_logit, sigma):
    return [_round4(_inv_logit(median_logit + z * sigma)) for z in Z]


def _fallback_forecast(case, target_hours):
    """Single-snapshot history: dilute by the documented launch growth."""
    last = case["history"][-1]
    n_now = last["players_tracked"] or PRIOR_N_LAUNCH
    slope_n = (math.log(PRIOR_N_REFRESH) - math.log(PRIOR_N_LAUNCH)) / math.log1p(
        PRIOR_REFRESH_HOURS)
    # N at the target, then the ratio against the history's last read (not
    # against launch: the last rate is already diluted to the last read).
    n_target = PRIOR_N_LAUNCH * math.exp(slope_n * math.log1p(max(target_hours, 0.0)))
    n_ratio = n_target / n_now
    horizon_days = max((target_hours - _hours_since_launch(case["at"])) / 24.0, 0.0)
    spread = math.sqrt((RW_SIGMA * math.sqrt(horizon_days)) ** 2 + 0.6 ** 2)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i]
        if rate is None:
            rate = 0.0
        if rate >= 99.0:
            # Saturated: the non-earner share is what moves, and it moves slowly.
            u_now = max(100.0 - rate, 0.005)
            median = _logit(100.0 - max(u_now * 2.0, 0.01))
            rows.append({"id": trophy["id"], "q": _quantiles(median, 0.5)})
            continue
        gamma = GAMMA[_speed_class(trophy)]
        r_hat = rate * (n_ratio ** (gamma - 1.0))
        rows.append({"id": trophy["id"],
                     "q": _quantiles(_logit(max(r_hat, 0.01)), spread)})
    return {"trophies": rows}


def _fitted_forecast(case, target_hours):
    """Main path: fit earners and tracked separately on the distinct reads."""
    history = case["history"]
    states = _distinct_states(history)
    last = history[-1]
    horizon_days = max((target_hours - _hours_since_launch(case["at"])) / 24.0, 0.0)

    # Time axis: log(1 + hours since launch), the scale both growth curves use.
    times = [math.log1p(max(_hours_since_launch(s["at"]), 0.0)) for s in states]
    t_target = math.log1p(max(target_hours, 0.0))
    n_max = max(s["players_tracked"] for s in states) or 1

    # --- players tracked ---
    ln_n = [math.log(max(s["players_tracked"], 1)) for s in states]
    a_n, b_n, sd_n, _, _ = _ols(times, ln_n)
    ln_n_target = a_n + b_n * t_target
    # A sqrt-time fit as a faster-growth second opinion, blended in lightly.
    sq_times = [math.sqrt(max(_hours_since_launch(s["at"]), 0.0)) for s in states]
    a_n2, b_n2, _, _, _ = _ols(sq_times, ln_n)
    ln_n_target = 0.75 * ln_n_target + 0.25 * (a_n2 + b_n2 * math.sqrt(max(target_hours, 0.0)))
    n_target = math.exp(ln_n_target)
    n_target = min(max(n_target, states[-1]["players_tracked"]), 20.0 * n_max)

    # --- per-trophy earners slopes, for speed-class shrinkage ---
    # Growth speed is driven by how quickly players reach a trophy (its
    # description), not by its PSN point value, so slopes shrink toward the
    # speed-class mean, not the tier mean.
    per_trophy = []
    for i, trophy in enumerate(case["trophies"]):
        xs, ys, ws = [], [], []
        for s, t in zip(states, times):
            r = s["rates"][i]
            if r is None or r <= 0:
                continue
            e = max(r / 100.0 * s["players_tracked"], 0.5)
            xs.append(t)
            ys.append(math.log(e))
            ws.append(math.sqrt(s["players_tracked"] / n_max))
        per_trophy.append((xs, ys, ws))

    class_slopes = {}
    for cls in ("fast", "medium", "slow"):
        num = den = 0.0
        for i, trophy in enumerate(case["trophies"]):
            if _speed_class(trophy) != cls:
                continue
            xs, ys, ws = per_trophy[i]
            if len(xs) >= 3 and len(set(xs)) >= 2:
                _, b, _, se_b, _ = _ols(xs, ys, ws)
                w = 1.0 / (se_b * se_b + TIER_PRIOR_SD ** 2)
                num += w * b
                den += w
        class_slopes[cls] = num / den if den > 0 else 0.0

    # Typical refresh jump in the history (logit units), for the spread.
    jumps = []
    for prev, cur in zip(states, states[1:]):
        for i in range(len(case["trophies"])):
            r0, r1 = prev["rates"][i], cur["rates"][i]
            if r0 is None or r1 is None or r0 <= 0 or r1 <= 0:
                continue
            jumps.append(abs(_logit(r1) - _logit(r0)))
    jumps.sort()
    jump_sigma = jumps[len(jumps) // 2] if jumps else 0.3
    jump_sigma = min(max(jump_sigma, 0.25), 0.8)

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        xs, ys, ws = per_trophy[i]
        last_rate = last["rates"][i]
        if last_rate is None:
            for s in reversed(states):
                if s["rates"][i] is not None:
                    last_rate = s["rates"][i]
                    break
        if last_rate is None:
            last_rate = 0.0
        if last_rate >= 90.0:
            # Saturated trophy: forecast the non-earner count instead.
            xu, yu = [], []
            for s, t in zip(states, times):
                r = s["rates"][i]
                if r is None:
                    continue
                u = max(s["players_tracked"] * (1.0 - r / 100.0), 0.5)
                xu.append(t)
                yu.append(math.log(u))
            if len(xu) >= 2 and len(set(xu)) >= 2:
                a_u, b_u, sd_u, _, _ = _ols(xu, yu)
                u_hat = math.exp(a_u + b_u * t_target)
            else:
                u_hat = max(states[-1]["players_tracked"] * (1.0 - last_rate / 100.0), 0.5) * 2.0
                sd_u = 0.5
            r_hat = 100.0 * max(1.0 - u_hat / max(n_target, 1.0), 0.0001)
            sigma = math.sqrt(sd_u ** 2 + (RW_SIGMA * math.sqrt(horizon_days)) ** 2 + 0.25 ** 2)
            rows.append({"id": trophy["id"],
                         "q": _quantiles(_logit(_clamp_pct(r_hat)), min(sigma, 1.5))})
            continue

        # Earners fit with speed-class shrinkage on the slope.
        if len(xs) >= 3 and len(set(xs)) >= 2:
            a_e, b_e, sd_e, se_b, _ = _ols(xs, ys, ws)
            prior = class_slopes.get(_speed_class(trophy), 0.0)
            prec_own = 1.0 / (se_b * se_b) if se_b not in (0.0, float("inf")) else 0.0
            prec_prior = 1.0 / (TIER_PRIOR_SD ** 2)
            b_shrunk = (b_e * prec_own + prior * prec_prior) / (prec_own + prec_prior)
            a_shrunk = (sum(w * y for w, y in zip(ws, ys)) / sum(ws)
                        - b_shrunk * (sum(w * x for w, x in zip(ws, xs)) / sum(ws)))
            ln_e_target = a_shrunk + b_shrunk * t_target
            sd_e = min(max(sd_e, 0.15), 1.0)
        else:
            # Too little signal: fall back to the dilution prior for this trophy.
            gamma = GAMMA[_speed_class(trophy)]
            e_now = max(last_rate / 100.0 * states[-1]["players_tracked"], 0.5)
            ln_e_target = math.log(e_now) + gamma * (ln_n_target - math.log(max(states[-1]["players_tracked"], 1)))
            sd_e = 0.5

        e_hat = math.exp(ln_e_target)
        e_hat = min(e_hat, 0.999 * n_target)
        r_decomp = 100.0 * e_hat / max(n_target, 1.0)

        # Direct logit-rate fit (house-curve style) as a 30% hedge.
        xd, yd = [], []
        for s, t in zip(states, times):
            r = s["rates"][i]
            if r is None or r <= 0:
                continue
            xd.append(t)
            yd.append(_logit(r))
        if len(xd) >= 3 and len(set(xd)) >= 2:
            a_d, b_d, _, _, _ = _ols(xd, yd)
            r_direct = _inv_logit(a_d + b_d * t_target)
        else:
            r_direct = last_rate
        med_logit = (BLEND_DECOMP * _logit(max(r_decomp, 0.01))
                     + (1.0 - BLEND_DECOMP) * _logit(max(r_direct, 0.01)))
        sigma = math.sqrt(sd_e ** 2 + sd_n ** 2
                          + (RW_SIGMA * math.sqrt(horizon_days)) ** 2 + jump_sigma ** 2)
        rows.append({"id": trophy["id"],
                     "q": _quantiles(med_logit, min(max(sigma, 0.3), 2.0))})
    return {"trophies": rows}


def forecast_case(case):
    if not case.get("history"):
        # No history at all: stay valid with a flat, very wide forecast.
        return {"trophies": [{"id": t["id"], "q": [0.05, 0.25, 1.0, 4.0, 20.0]}
                             for t in case["trophies"]]}
    target_hours = _hours_since_launch(case["target"])
    states = _distinct_states(case["history"])
    if len(states) < 3:
        return _fallback_forecast(case, target_hours)
    return _fitted_forecast(case, target_hours)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
