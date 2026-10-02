from src.eval.decision_model import build_tier_profiles, choose_route_by_ev, expected_value


def _record(score, naive=None, parent=None, hyde=None):
    default = {"cost_usd": 0.0, "latency_ms": 0.0, "faithfulness": 1.0, "relevancy": 1.0}
    return {
        "complexity_score": score,
        "strategies": {
            "naive": {**default, **(naive or {})},
            "parent": {**default, **(parent or {})},
            "hyde": {**default, **(hyde or {})},
        },
    }


def test_expected_value_subtracts_cost_and_latency():
    cell = {"quality": 1.0, "cost_usd": 0.1, "latency_ms": 2000}
    # 1.0 - (10 * 0.1) - (0.5 * 2.0) = 1.0 - 1.0 - 1.0 = -1.0
    assert expected_value(cell, lambda_cost=10, lambda_latency=0.5) == -1.0


def test_build_tier_profiles_omits_cells_below_min_samples():
    records = [_record(0.1), _record(0.2)]  # only 2 easy-band records
    profiles = build_tier_profiles(records, min_samples=3)
    assert "naive" not in profiles  # easy band has only 2 samples, below min_samples=3
    assert "global" not in profiles  # global also only has 2 samples


def test_build_tier_profiles_computes_band_and_global_stats():
    records = [_record(0.1, naive={"cost_usd": 0.01}) for _ in range(3)]
    profiles = build_tier_profiles(records, min_samples=3)
    assert profiles["naive"]["naive"]["n"] == 3
    assert profiles["naive"]["naive"]["cost_usd"] == 0.01
    assert profiles["global"]["naive"]["n"] == 3


def test_choose_route_by_ev_uses_band_level_data_when_available():
    # Easy-band naive tier is terrible (low quality), parent tier is great — EV should prefer parent.
    records = [
        _record(0.1, naive={"faithfulness": 0.0, "relevancy": 0.0}, parent={"faithfulness": 1.0, "relevancy": 1.0})
        for _ in range(3)
    ]
    profiles = build_tier_profiles(records, min_samples=3)
    route, used_ev, ev_scores = choose_route_by_ev(
        0.1, profiles, lambda_cost=0.0, lambda_latency=0.0, t1=0.4, t2=0.85
    )
    assert used_ev is True
    assert route == "parent"
    assert ev_scores["parent"] > ev_scores["naive"]


def test_choose_route_by_ev_falls_back_to_global_when_band_sparse():
    # Only 3 total records, none fall in the "hard" band (score 0.1 -> "naive" band),
    # so looking up a hard-band score (0.9) must fall back to the global pool.
    records = [_record(0.1) for _ in range(3)]
    profiles = build_tier_profiles(records, min_samples=3)
    route, used_ev, ev_scores = choose_route_by_ev(
        0.9, profiles, lambda_cost=0.0, lambda_latency=0.0, t1=0.4, t2=0.85
    )
    assert used_ev is True  # global pool has 3 samples per tier, so EV still fires
    assert ev_scores is not None


def test_choose_route_by_ev_falls_back_to_threshold_when_no_data_anywhere():
    route, used_ev, ev_scores = choose_route_by_ev(
        0.9, profiles={}, lambda_cost=0.0, lambda_latency=0.0, t1=0.4, t2=0.85
    )
    assert used_ev is False
    assert ev_scores is None
    assert route == "hyde"  # score_to_tier(0.9, 0.4, 0.85) == "hyde"
