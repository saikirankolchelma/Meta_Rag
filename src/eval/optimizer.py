"""Bayesian (Optuna) search over routing thresholds T1/T2 that minimizes total
cost subject to an average-accuracy SLA, using the shadow harness's brute-force
results. Pushes the winning thresholds to Redis, where the live gateway can
read them without a restart.

Usage:
    python -m src.eval.optimizer [--trials 50] [--sla 0.90]
"""

import argparse
import json

import optuna

from src.utils.cache import get_client as get_redis_client
from src.utils.config import DATA_PROCESSED_DIR

RESULTS_PATH = DATA_PROCESSED_DIR / "shadow_eval_results.jsonl"
REDIS_KEY_T1 = "router:threshold:T1"
REDIS_KEY_T2 = "router:threshold:T2"

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_records() -> list[dict]:
    if not RESULTS_PATH.exists():
        raise SystemExit(f"{RESULTS_PATH} not found. Run shadow_harness.py first.")
    return [json.loads(line) for line in RESULTS_PATH.open(encoding="utf-8")]


def _strategy_for_score(score: float, t1: float, t2: float) -> str:
    if score < t1:
        return "naive"
    if score < t2:
        return "parent"
    return "hyde"


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


def push_thresholds(t1: float, t2: float) -> None:
    client = get_redis_client()
    client.set(REDIS_KEY_T1, t1)
    client.set(REDIS_KEY_T2, t2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--sla", type=float, default=0.90)
    args = parser.parse_args()

    records = load_records()
    print(f"Loaded {len(records)} shadow-eval records.")

    result = optimize(records, args.trials, args.sla)
    if not result["met_sla"]:
        print(
            f"WARNING: no threshold split reached the {args.sla:.0%} accuracy SLA. "
            f"Using the best available split instead (accuracy={result['accuracy']:.3f})."
        )

    print(
        f"Best thresholds: T1={result['T1']:.3f} T2={result['T2']:.3f} "
        f"cost=${result['cost']:.4f} accuracy={result['accuracy']:.3f}"
    )

    push_thresholds(result["T1"], result["T2"])
    print(f"Pushed thresholds to Redis ({REDIS_KEY_T1}, {REDIS_KEY_T2}).")


if __name__ == "__main__":
    main()
