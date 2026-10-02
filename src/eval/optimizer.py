"""Bayesian (Optuna) search over the routing decision model, using the shadow
harness's brute-force results. Two models are tuned:

- `optimize()` / `push_thresholds()`: the original 2-parameter (T1, T2)
  static-threshold rule — kept as a baseline for an "EV vs. pure-threshold"
  cost/accuracy comparison, and because T1/T2 remain the real cold-start
  fallback values for a fresh deployment with no shadow-eval history yet.
- `optimize_ev()` / `push_decision_config()`: the EV decision model (see
  decision_model.py) — tunes T1/T2 (fallback) plus lambda_cost/lambda_latency
  (the EV weights), and pushes the whole decision config as one Redis key so
  the live gateway reads it without a restart.

Usage:
    python -m src.eval.optimizer [--trials 150] [--sla 0.90]
"""

import argparse
import json
import time

import optuna

from src.eval.decision_model import build_tier_profiles, choose_route_by_ev
from src.eval.routing_rules import score_to_tier
from src.utils.cache import get_client as get_redis_client
from src.utils.config import DATA_PROCESSED_DIR

RESULTS_PATH = DATA_PROCESSED_DIR / "shadow_eval_results.jsonl"
REDIS_KEY_T1 = "router:threshold:T1"
REDIS_KEY_T2 = "router:threshold:T2"
REDIS_KEY_DECISION_CONFIG = "router:decision_config"

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_records() -> list[dict]:
    if not RESULTS_PATH.exists():
        raise SystemExit(f"{RESULTS_PATH} not found. Run shadow_harness.py first.")
    return [json.loads(line) for line in RESULTS_PATH.open(encoding="utf-8")]


# Re-exported, not removed: tests/test_optimizer.py imports this name directly.
_strategy_for_score = score_to_tier


def simulate_routing(records: list[dict], t1: float, t2: float) -> tuple[float, float]:
    total_cost = 0.0
    accuracies = []
    for record in records:
        strategy = _strategy_for_score(record["complexity_score"], t1, t2)
        result = record["strategies"][strategy]
        total_cost += result["cost_usd"]
        accuracies.append((result["faithfulness"] + result["relevancy"]) / 2)
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    return total_cost, avg_accuracy


def simulate_routing_with_ev(
    records: list[dict], profiles: dict, t1: float, t2: float, lambda_cost: float, lambda_latency: float
) -> tuple[float, float]:
    total_cost = 0.0
    accuracies = []
    for record in records:
        route, _, _ = choose_route_by_ev(record["complexity_score"], profiles, lambda_cost, lambda_latency, t1, t2)
        result = record["strategies"][route]
        total_cost += result["cost_usd"]
        accuracies.append((result["faithfulness"] + result["relevancy"]) / 2)
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    return total_cost, avg_accuracy


def optimize(records: list[dict], n_trials: int, sla: float) -> dict:
    def objective(trial: optuna.Trial) -> float:
        t1 = trial.suggest_float("T1", 0.1, 0.6)
        t2 = trial.suggest_float("T2", 0.5, 0.95)
        if t1 >= t2:
            trial.set_user_attr("accuracy", 0.0)
            trial.set_user_attr("cost", float("inf"))
            return float("inf")

        cost, accuracy = simulate_routing(records, t1, t2)
        trial.set_user_attr("accuracy", accuracy)
        trial.set_user_attr("cost", cost)
        if accuracy < sla:
            return float("inf")
        return cost

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    feasible = [t for t in study.trials if t.value is not None and t.value != float("inf")]
    if feasible:
        best = min(feasible, key=lambda t: t.value)
        met_sla = True
    else:
        # No threshold split hit the SLA in any trial; fall back to whichever
        # split achieved the highest accuracy instead of failing outright.
        best = max(study.trials, key=lambda t: t.user_attrs.get("accuracy", 0.0))
        met_sla = False

    return {
        "T1": best.params["T1"],
        "T2": best.params["T2"],
        "cost": best.user_attrs["cost"],
        "accuracy": best.user_attrs["accuracy"],
        "met_sla": met_sla,
    }


def optimize_ev(records: list[dict], n_trials: int, sla: float) -> dict:
    # Profiles depend only on the dataset and the fixed band boundaries, not
    # on any trial parameter — building them once avoids redoing this work
    # on every one of n_trials objective evaluations.
    profiles = build_tier_profiles(records)

    def objective(trial: optuna.Trial) -> float:
        t1 = trial.suggest_float("T1", 0.1, 0.6)
        t2 = trial.suggest_float("T2", 0.5, 0.95)
        # Range calibrated to the ~$0.05 max cost delta between tiers in this
        # dataset (1/0.05=20); past ~30-40 hyde is vetoed regardless of
        # quality, so most of a wider range like [0,200] would be wasted.
        lambda_cost = trial.suggest_float("lambda_cost", 0.0, 40.0)
        # Range calibrated to the ~1-25s latency spread (1/25=0.04).
        lambda_latency = trial.suggest_float("lambda_latency", 0.0, 0.1)
        if t1 >= t2:
            trial.set_user_attr("accuracy", 0.0)
            trial.set_user_attr("cost", float("inf"))
            return float("inf")

        cost, accuracy = simulate_routing_with_ev(records, profiles, t1, t2, lambda_cost, lambda_latency)
        trial.set_user_attr("accuracy", accuracy)
        trial.set_user_attr("cost", cost)
        if accuracy < sla:
            return float("inf")
        return cost

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    feasible = [t for t in study.trials if t.value is not None and t.value != float("inf")]
    if feasible:
        best = min(feasible, key=lambda t: t.value)
        met_sla = True
    else:
        best = max(study.trials, key=lambda t: t.user_attrs.get("accuracy", 0.0))
        met_sla = False

    return {
        "T1": best.params["T1"],
        "T2": best.params["T2"],
        "lambda_cost": best.params["lambda_cost"],
        "lambda_latency": best.params["lambda_latency"],
        "cost": best.user_attrs["cost"],
        "accuracy": best.user_attrs["accuracy"],
        "met_sla": met_sla,
        "profiles": profiles,
    }


def push_thresholds(t1: float, t2: float) -> None:
    client = get_redis_client()
    client.set(REDIS_KEY_T1, t1)
    client.set(REDIS_KEY_T2, t2)


def push_decision_config(t1: float, t2: float, lambda_cost: float, lambda_latency: float, profiles: dict) -> None:
    # One consolidated key, one SET — atomic, unlike five independent writes
    # (which could let a reader observe new profiles paired with stale
    # lambdas mid-update) — and it's one Redis round-trip per gateway request
    # instead of up to five.
    payload = {
        "t1": t1,
        "t2": t2,
        "lambda_cost": lambda_cost,
        "lambda_latency": lambda_latency,
        "profiles": profiles,
        "generated_at": time.time(),
    }
    client = get_redis_client()
    client.set(REDIS_KEY_DECISION_CONFIG, json.dumps(payload))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=150)
    parser.add_argument("--sla", type=float, default=0.90)
    args = parser.parse_args()

    records = load_records()
    print(f"Loaded {len(records)} shadow-eval records.")

    baseline = optimize(records, args.trials, args.sla)
    print(
        f"[baseline: static threshold] T1={baseline['T1']:.3f} T2={baseline['T2']:.3f} "
        f"cost=${baseline['cost']:.4f} accuracy={baseline['accuracy']:.3f} met_sla={baseline['met_sla']}"
    )

    result = optimize_ev(records, args.trials, args.sla)
    if not result["met_sla"]:
        print(
            f"WARNING: no EV configuration reached the {args.sla:.0%} accuracy SLA. "
            f"Using the best available configuration instead (accuracy={result['accuracy']:.3f})."
        )

    print(
        f"[EV decision model]        T1={result['T1']:.3f} T2={result['T2']:.3f} "
        f"lambda_cost={result['lambda_cost']:.3f} lambda_latency={result['lambda_latency']:.4f} "
        f"cost=${result['cost']:.4f} accuracy={result['accuracy']:.3f}"
    )
    if baseline["cost"] > 0:
        savings_pct = (1 - result["cost"] / baseline["cost"]) * 100
        direction = "lower" if savings_pct >= 0 else "higher"
        print(
            f"EV vs. static threshold: {abs(savings_pct):.1f}% {direction} cost "
            f"(accuracy {result['accuracy']:.3f} vs. baseline {baseline['accuracy']:.3f})."
        )

    push_decision_config(result["T1"], result["T2"], result["lambda_cost"], result["lambda_latency"], result["profiles"])
    print(f"Pushed decision config to Redis ({REDIS_KEY_DECISION_CONFIG}).")


if __name__ == "__main__":
    main()
