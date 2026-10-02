"""Verify the paper's core contrasts from the small frozen result tables.

This script checks reported arithmetic and rebuilds the paired two-day
bootstrap intervals from released day-level aggregates. It does not rerun
traffic-speed model training or reconstruct the proxy network from raw data.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CHECKS = 0


def rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def near(label: str, actual: float, expected: float, tol: float = 1e-5) -> None:
    global CHECKS
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=tol):
        raise AssertionError(f"{label}: {actual:.12g} != {expected:.12g}")
    CHECKS += 1


def check_scope() -> None:
    scope = rows("dataset_scope.csv")
    if {r["city"] for r in scope} != {"Beijing", "Chengdu"}:
        raise AssertionError("Expected Beijing and Chengdu scope rows")
    for r in scope:
        # departure_times counts all departures across the 14 test dates.
        n = int(r["od_pairs"]) * int(r["departure_times"])
        near(f"{r['city']} repeated observations", n,
             int(r["decision_samples_per_configuration"]), 0)
        near(f"{r['city']} candidate paths", int(r["paths_per_od"]), 3, 0)
        near(f"{r['city']} test dates", int(r["test_dates"]), 14, 0)
        daily = int(r["departure_times"]) / int(r["test_dates"])
        near(f"{r['city']} departures per test day", daily,
             56 if r["city"] == "Beijing" else 25, 0)


def check_accuracy() -> None:
    source = {(r["城市"], r["模型"]): r for r in rows("accuracy_decision.csv")}
    m0, m2 = source[("成都", "M0")], source[("成都", "M2")]
    mae_reduction = 100 * (1 - float(m2["测试速度MAE_km_h"]) /
                           float(m0["测试速度MAE_km_h"]))
    late_gain_pp = float(m2["C0迟到率_百分点"]) - float(m0["C0迟到率_百分点"])
    time_gain_min = (float(m2["C0平均旅行时间_分钟"]) -
                     float(m0["C0平均旅行时间_分钟"]))
    near("Chengdu M2 speed MAE reduction (%)", mae_reduction, 14.91, 0.01)
    near("Chengdu M2-M0 C0 lateness (pp)", late_gain_pp, -1.4571, 0.0001)
    near("Chengdu M2-M0 C0 time (min)", time_gain_min, -0.1250, 0.0001)


def noncircular_block2_indices() -> np.ndarray:
    with np.load(DATA / "beijing_bootstrap_indices.npz") as saved:
        if int(saved["seed"]) != 20260920:
            raise AssertionError("Unexpected Beijing bootstrap seed")
        return saved["block2"].copy()


def circular_block2_indices(n: int = 14, draws: int = 2000,
                            seed: int = 20260924) -> np.ndarray:
    # Exact resampling algorithm used for the frozen Chengdu formal run.
    rng = np.random.default_rng(seed)
    output = np.empty((draws, n), dtype=np.int16)
    for draw in range(draws):
        sampled: list[int] = []
        for _ in range(math.ceil(n / 2)):
            start = int(rng.integers(0, n))
            sampled.extend(((start + offset) % n) for offset in range(2))
        output[draw] = sampled[:n]
    return output


def interval(values: np.ndarray, indices: np.ndarray,
             unit: float = 1.0) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (14,) or indices.shape != (2000, 14):
        raise AssertionError("Expected 14 dates and 2,000 bootstrap draws")
    boot = values[indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return tuple(float(x) * unit for x in (values.mean(), low, high))


def check_primary() -> None:
    primary = {r["city_model"]: r for r in rows("primary_comparisons.csv")}
    if set(primary) != {"Beijing M1", "Chengdu M1", "Chengdu M2"}:
        raise AssertionError("Unexpected primary-comparison set")
    bridge = rows("switch_decomposition.csv")
    # The two Chengdu rows share the same branch label; distinguish by model.
    switch = {
        "Beijing M1": next(r for r in bridge if "three-seed pooled" in r["branch"]),
        "Chengdu M1": next(r for r in bridge if r["city_model"] == "Chengdu M1"),
        "Chengdu M2": next(r for r in bridge if r["city_model"] == "Chengdu M2"),
    }
    for name, r in primary.items():
        coverage_c0 = float(r["joint_coverage_c0_pct"])
        coverage_c2 = float(r["joint_coverage_c2_pct"])
        near(f"{name} C0 joint deviation", abs(coverage_c0 - 90),
             float(r["joint_abs_deviation_c0_pp"]))
        near(f"{name} C2 joint deviation", abs(coverage_c2 - 90),
             float(r["joint_abs_deviation_c2_pp"]))
        near(f"{name} upper-score expansion",
             float(r["c2_u_minus_q50_candidate_mean_min"]) -
             float(r["raw_q90_minus_q50_candidate_mean_min"]),
             float(r["c2_u_minus_q90_candidate_mean_min"]))
        near(f"{name} extra late per 10,000",
             100 * float(r["lateness_change_pp"]),
             float(r["extra_late_per_10000"]))
        b = switch[name]
        net = int(b["harmful_switches"]) - int(b["beneficial_switches"])
        n = int(b["decision_samples"])
        near(f"{name} switch decomposition", net,
             int(b["net_harmful_switches"]), 0)
        near(f"{name} lateness from switches", 100 * net / n,
             float(r["lateness_change_pp"]))
        near(f"{name} route-change rate", 100 * int(b["route_changes"]) / n,
             float(r["route_change_rate_pct"]))
        near(f"{name} travel time CI sign", float(r["travel_time_ci95_low_sec"]) > 0,
             1, 0)
        near(f"{name} lateness CI sign", float(r["lateness_ci95_low_pp"]) > 0,
             1, 0)
    for b in bridge:
        near(f"{b['city_model']} {b['branch']} net switches",
             int(b["harmful_switches"]) - int(b["beneficial_switches"]),
             int(b["net_harmful_switches"]), 0)
    near("Beijing formal net switches across seeds",
         sum(int(b["net_harmful_switches"]) for b in bridge
             if b["branch"].startswith("S6 formal B=500, seed")), 576, 0)
    near("Beijing supplement separate net switches",
         int(next(b for b in bridge if "E168-CDF" in b["branch"])["net_harmful_switches"]),
         155, 0)

    bj_daily = sorted((r for r in rows("beijing_daily_paired.csv")
                       if r["model"] == "M1"), key=lambda r: int(r["day"]))
    if len(bj_daily) != 14:
        raise AssertionError("Beijing M1 should have 14 daily paired rows")
    bj_idx = noncircular_block2_indices()
    for metric, factor, estimate_col, low_col, high_col in (
        ("late_diff", 100, "lateness_change_pp", "lateness_ci95_low_pp", "lateness_ci95_high_pp"),
        ("travel_diff", 60, "travel_time_change_sec", "travel_time_ci95_low_sec", "travel_time_ci95_high_sec"),
    ):
        e, lo, hi = interval(np.array([float(r[metric]) for r in bj_daily]), bj_idx, factor)
        for label, actual, column in (("estimate", e, estimate_col),
                                       ("lower", lo, low_col), ("upper", hi, high_col)):
            near(f"Beijing {metric} {label}", actual, float(primary["Beijing M1"][column]))

    cd_daily = rows("chengdu_daily_metrics.csv")
    cd_idx = circular_block2_indices()
    for model in ("M1", "M2"):
        subset = {}
        for method in ("C0", "C2"):
            group = sorted((r for r in cd_daily if r["model"] == model
                            and r["method"] == method), key=lambda r: r["date"])
            if len(group) != 14:
                raise AssertionError(f"Chengdu {model}/{method}: expected 14 dates")
            subset[method] = group
        target = primary[f"Chengdu {model}"]
        for metric, factor, estimate_col, low_col, high_col in (
            ("late_rate", 100, "lateness_change_pp", "lateness_ci95_low_pp", "lateness_ci95_high_pp"),
            ("mean_travel_time_min", 60, "travel_time_change_sec", "travel_time_ci95_low_sec", "travel_time_ci95_high_sec"),
        ):
            values = np.array([float(a[metric]) - float(b[metric])
                               for a, b in zip(subset["C2"], subset["C0"])])
            e, lo, hi = interval(values, cd_idx, factor)
            for label, actual, column in (("estimate", e, estimate_col),
                                           ("lower", lo, low_col), ("upper", hi, high_col)):
                near(f"Chengdu {model} {metric} {label}", actual, float(target[column]))


def check_interfaces() -> None:
    interfaces = rows("decision_interfaces.csv")
    formal = {r["strategy"]: r for r in interfaces if r["city"] == "Chengdu"}
    for strategy, value in (("M2/P-Q_C0", 17.4705), ("M2/P-Q_C2", 18.3905),
                            ("M2/SCENARIO-MEAN", 16.8095), ("ORACLE", 12.8305)):
        near(f"Chengdu interface {strategy}", float(formal[strategy]["lateness_rate_pct"]),
             value, 0.0001)
    beijing = [r for r in interfaces if r["city"] == "Beijing"]
    if not beijing or not all("supplemental" in r["validation_status"].lower()
                              or "not deployable" in r["validation_status"].lower()
                              for r in beijing):
        raise AssertionError("Beijing interface branch must remain supplemental")


def main() -> None:
    check_scope()
    check_accuracy()
    check_primary()
    check_interfaces()
    print(f"PASS: {CHECKS} core arithmetic and interval checks")
    print("Beijing formal and E168-CDF supplemental branches remain separate.")
    print("These checks begin from frozen daily aggregates, not raw speed records.")


if __name__ == "__main__":
    main()
