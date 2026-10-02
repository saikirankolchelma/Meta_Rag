"""Expected-value (EV) tier selection, built from shadow-eval history.

Replaces "which band does this score fall in" with "which tier actually
delivers the best quality-minus-cost-minus-latency tradeoff for queries like
this one" — using empirical stats from judge-scored shadow-eval records
rather than a trained regression model, since the available sample sizes
(dozens, not thousands) would make a regression overfit and harder to defend
than robust per-band averages.

Historical records are grouped into "bands" using the FIXED T1_DEFAULT/
T2_DEFAULT (not the live, nightly-retuned T1/T2) — band membership must stay
stable across optimizer runs, or "accumulate enough samples per cell over
time" never converges. A "global" band (all records, pooled) backstops any
band that doesn't yet have enough data for all three tiers, so EV can fire
on effectively all traffic from day one rather than only once per-band data
piles up. Only a tier with literally zero historical data anywhere falls
through to the plain score_to_tier threshold rule.

Known limitation: generate_groq's cost is hardcoded to $0.0 (the Groq free
tier isn't metered), but it isn't actually free — it shares a real, scarce
daily token quota (see dispatcher.py's note on groq/compound-mini's hidden
cap). Because judge-scored quality is frequently tied near 1.0 across tiers
in this dataset, EV will often prefer "parent" over "naive" on cost alone
once quality ties, which could drain that shared quota faster than the old
static-threshold rule did. Not fixed here — documented as a known tradeoff.
"""

from src.eval.routing_rules import T1_DEFAULT, T2_DEFAULT, score_to_tier

TIERS = ("naive", "parent", "hyde")
BANDS = ("naive", "parent", "hyde", "global")  # band names reuse tier names; "global" is pooled


def build_tier_profiles(records: list[dict], min_samples: int = 3) -> dict:
    """Returns {band: {tier: {"quality", "cost_usd", "latency_ms", "n"}}},
    omitting any band/tier cell with fewer than min_samples records."""
    buckets: dict[str, dict[str, list[dict]]] = {band: {tier: [] for tier in TIERS} for band in BANDS}

    for record in records:
        band = score_to_tier(record["complexity_score"], T1_DEFAULT, T2_DEFAULT)
        for tier in TIERS:
            result = record["strategies"][tier]
            buckets[band][tier].append(result)
            buckets["global"][tier].append(result)

    profiles: dict[str, dict[str, dict]] = {}
    for band, tiers in buckets.items():
        for tier, results in tiers.items():
            if len(results) < min_samples:
                continue
            n = len(results)
            quality = sum((r["faithfulness"] + r["relevancy"]) / 2 for r in results) / n
            cost_usd = sum(r["cost_usd"] for r in results) / n
            latency_ms = sum(r["latency_ms"] for r in results) / n
            profiles.setdefault(band, {})[tier] = {
                "quality": quality,
                "cost_usd": cost_usd,
                "latency_ms": latency_ms,
                "n": n,
            }
    return profiles


def expected_value(cell: dict, lambda_cost: float, lambda_latency: float) -> float:
    return cell["quality"] - lambda_cost * cell["cost_usd"] - lambda_latency * (cell["latency_ms"] / 1000)


def choose_route_by_ev(
    score: float,
    profiles: dict,
    lambda_cost: float,
    lambda_latency: float,
    t1: float,
    t2: float,
) -> tuple[str, bool, dict | None]:
    """Returns (route, used_ev, ev_scores). ev_scores is None when the plain
    threshold fallback was used (no historical data at band or global level)."""
    band = score_to_tier(score, T1_DEFAULT, T2_DEFAULT)

    for candidate_band in (band, "global"):
        cells = profiles.get(candidate_band, {})
        if all(tier in cells for tier in TIERS):
            ev_scores = {tier: expected_value(cells[tier], lambda_cost, lambda_latency) for tier in TIERS}
            best_tier = max(ev_scores, key=ev_scores.get)
            return best_tier, True, ev_scores

    return score_to_tier(score, t1, t2), False, None
