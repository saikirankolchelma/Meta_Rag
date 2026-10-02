from src.eval.routing_rules import score_to_tier


def test_score_below_t1_is_naive():
    assert score_to_tier(0.05, t1=0.4, t2=0.85) == "naive"
    assert score_to_tier(0.39, t1=0.4, t2=0.85) == "naive"


def test_score_between_t1_and_t2_is_parent():
    assert score_to_tier(0.4, t1=0.4, t2=0.85) == "parent"  # t1 boundary is inclusive of parent
    assert score_to_tier(0.6, t1=0.4, t2=0.85) == "parent"
    assert score_to_tier(0.84, t1=0.4, t2=0.85) == "parent"


def test_score_at_or_above_t2_is_hyde():
    assert score_to_tier(0.85, t1=0.4, t2=0.85) == "hyde"
    assert score_to_tier(0.99, t1=0.4, t2=0.85) == "hyde"
