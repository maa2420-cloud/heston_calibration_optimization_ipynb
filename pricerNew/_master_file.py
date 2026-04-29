import os
import pandas as pd
import numpy as np

import _baseline as _baseline
import _improved as improved

valid_strike = [30, 40, 60, 80, 90, 95, 97.5, 100, 105, 110, 120, 130, 150, 300]
valid_tenor = ["1W", "2W", "3W", "1M", "2M", "3M", "6M", "9M", "1Y", "18M", "2Y"]

def select_dates(number, cache_file="random_date.csv", regenerate=False):
    """
    Generate or load the shared date list. The two schemes will use identical dates
    so comparisons are paired.
    
    regenerate=True forces a fresh sample (deletes the cache).
    """
    if regenerate and os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"Removed cache {cache_file} to regenerate")

    if os.path.exists(cache_file):
        dates = pd.read_csv(cache_file, index_col=0).index.astype(str).tolist()
        print(f"Loaded {len(dates)} cached dates from {cache_file}")
        return dates

    # Need to construct candidate pool — replicate the filter logic ONCE here.
    df = pd.read_excel("OrganizedData.xlsx", index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")

    spot_df = pd.read_excel("SPY_Complete.xlsx", sheet_name="Underlying", index_col=0)
    other_df = pd.read_excel("SPY_Complete.xlsx", sheet_name="Other", index_col=0)
    # scores_df = pd.read_excel("standardized_scores.xlsx", index_col=0)

    df.index = df.index.astype(str)
    spot_df.index = spot_df.index.astype(str)
    other_df.index = other_df.index.astype(str)
    # scores_df.index = scores_df.index.astype(str)

    all_dates = df.index.intersection(spot_df.index.intersection(other_df.index))

    iv_cols = [
        f"{t}_{k}_Volatility"
        for t in _baseline.valid_tenor
        for k in _baseline.valid_strike
    ]

    # Filter must satisfy BOTH schemes — improved needs PCA scores, baseline doesn't,
    # but using the strict (PCA-required) filter is safe for both.
    all_dates = all_dates[
        spot_df.loc[all_dates, "Mid"].notna()
        & other_df.loc[all_dates, "Interest"].notna()
        & other_df.loc[all_dates, "Dividend"].notna()
        & df.loc[all_dates, iv_cols].notna().all(axis=1)
    #     & all_dates.isin(scores_df.index)
    ]

    # Restrict sampling to dates strictly after 2010-05-06.
    cutoff = pd.Timestamp("2010-05-06")
    parsed = pd.to_datetime(all_dates, errors="coerce")
    all_dates = all_dates[(parsed > cutoff) & parsed.notna()]
    print(f"Candidate pool after 2010-05-06: {len(all_dates)} dates")

    dates = sorted(np.random.choice(all_dates, size=number, replace=False).tolist())
    pd.DataFrame(index=dates).to_csv(cache_file)
    print(f"Generated {len(dates)} new random dates, cached to {cache_file}")
    return dates


def compare(number=20, regenerate_dates=False):
    """
    Run both schemes on identical dates and produce a paired comparison spreadsheet.
    """
    dates = select_dates(number, regenerate=regenerate_dates)

    print("\n" + "█" * 60)
    print(f"  RUNNING BASELINE SCHEME on {len(dates)} dates")
    print("█" * 60)
    _baseline.runSimulation(
        number=len(dates),
        random_dates=dates,
        output_file="simulation_results_baseline.xlsx"
    )

    print("\n" + "█" * 60)
    print(f"  RUNNING IMPROVED SCHEME on {len(dates)} dates")
    print("█" * 60)
    improved.runSimulation(
        number=len(dates),
        random_dates=dates,
        output_file="simulation_results_improved.xlsx"
    )

    print("\n" + "█" * 60)
    print("  BUILDING COMPARISON SPREADSHEET")
    print("█" * 60)
    build_comparison(
        baseline_file="simulation_results_baseline.xlsx",
        improved_file="simulation_results_improved.xlsx",
        output_file="comparison.xlsx",
    )


def build_comparison(baseline_file, improved_file, output_file):
    """Side-by-side comparison spreadsheet with paired metrics per date."""
    b = pd.read_excel(baseline_file, index_col=0)
    i = pd.read_excel(improved_file, index_col=0)

    # Align on date
    common = b.index.intersection(i.index)
    if len(common) < len(b) or len(common) < len(i):
        print(f"Warning: only {len(common)} dates in both files "
              f"(baseline: {len(b)}, improved: {len(i)})")
    b = b.loc[common]
    i = i.loc[common]

    # Build paired columns
    out = pd.DataFrame(index=common)
    metrics = ["initial_mse", "final_mse", "n_iterations",
               "n_function_evals", "cpu_time_s"]
    for m in metrics:
        if m in b.columns and m in i.columns:
            out[f"{m}_baseline"] = b[m]
            out[f"{m}_improved"] = i[m]
            # Ratios where meaningful
            if m in ("n_iterations", "n_function_evals", "cpu_time_s", "final_mse"):
                out[f"{m}_ratio"] = i[m] / b[m]

    # Calibrated parameters side-by-side
    for p in ["v0", "kappa", "theta", "eta", "rho"]:
        out[f"{p}_baseline"] = b[p]
        out[f"{p}_improved"] = i[p]

    out = out.sort_index()
    out.to_excel(output_file)

    # Headline summary
    print(f"\nComparison written to {output_file}")
    print("\n=== HEADLINE METRICS (median, paired) ===")
    for m in ["initial_mse", "final_mse", "n_iterations",
              "n_function_evals", "cpu_time_s"]:
        if f"{m}_baseline" in out.columns:
            mb = out[f"{m}_baseline"].median()
            mi = out[f"{m}_improved"].median()
            ratio = mi / mb if mb != 0 else float("nan")
            print(f"  {m:20s}  baseline={mb:>12.4f}  improved={mi:>12.4f}  "
                  f"ratio={ratio:.3f}")

    # Win count for the improved scheme
    print("\n=== PER-DATE WINS (improved < baseline) ===")
    for m in ["final_mse", "n_function_evals", "cpu_time_s"]:
        if f"{m}_baseline" in out.columns:
            wins = (out[f"{m}_improved"] < out[f"{m}_baseline"]).sum()
            total = len(out)
            print(f"  {m:20s}  improved better on {wins}/{total} dates "
                  f"({100*wins/total:.0f}%)")


if __name__ == "__main__":
    # Set regenerate_dates=True to force a fresh date sample
    compare(number=50, regenerate_dates=True)