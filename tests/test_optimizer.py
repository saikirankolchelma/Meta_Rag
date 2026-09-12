from src.eval.optimizer import _strategy_for_score, simulate_routing


def test_strategy_for_score_below_t1_is_naive():
    assert _strategy_for_score(0.05, t1=0.4, t2=0.85) == "naive"
    assert _strategy_for_score(0.39, t1=0.4, t2=0.85) == "naive"


def test_strategy_for_score_between_t1_and_t2_is_parent():
    assert _strategy_for_score(0.4, t1=0.4, t2=0.85) == "parent"  # t1 boundary is inclusive of parent
    assert _strategy_for_score(0.6, t1=0.4, t2=0.85) == "parent"
    assert _strategy_for_score(0.84, t1=0.4, t2=0.85) == "parent"


def test_strategy_for_score_at_or_above_t2_is_hyde():
    assert _strategy_for_score(0.85, t1=0.4, t2=0.85) == "hyde"
    assert _strategy_for_score(0.99, t1=0.4, t2=0.85) == "hyde"


def _record(score, naive=None, parent=None, hyde=None):
    default = {"cost_usd": 0.0, "faithfulness": 0.0, "relevancy": 0.0}
    return {
        "complexity_score": score,
        "strategies": {
            "naive": {**default, **(naive or {})},
            "parent": {**default, **(parent or {})},
            "hyde": {**default, **(hyde or {})},
        },
    }


def test_simulate_routing_sums_cost_only_for_the_chosen_strategy():
    records = [
        _record(0.1, naive={"cost_usd": 0.0}, hyde={"cost_usd": 999.0}),
        _record(0.9, parent={"cost_usd": 999.0}, hyde={"cost_usd": 0.02}),
    ]
    cost, _ = simulate_routing(records, t1=0.4, t2=0.85)
    # record 1 -> naive ($0), record 2 -> hyde ($0.02); the $999 decoys must be ignored
    assert cost == 0.02


def test_simulate_routing_averages_accuracy_across_records():
    records = [
        _record(0.1, naive={"faithfulness": 1.0, "relevancy": 1.0}),
        _record(0.9, hyde={"faithfulness": 0.0, "relevancy": 1.0}),
    ]
    _, accuracy = simulate_routing(records, t1=0.4, t2=0.85)
    # record 1 -> naive: (1.0+1.0)/2 = 1.0 ; record 2 -> hyde: (0.0+1.0)/2 = 0.5
    assert accuracy == (1.0 + 0.5) / 2


def test_simulate_routing_empty_records_returns_zero_cost_and_accuracy():
    cost, accuracy = simulate_routing([], t1=0.4, t2=0.85)
    assert cost == 0.0
    assert accuracy == 0.0
