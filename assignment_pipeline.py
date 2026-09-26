"""
Parts A, B, D, E — data loading, sampling, benchmarking, extrapolation.

HOW TO USE ON THE REAL DATASET:
  1. Get the data (see assignment PDF, Option A: Kaggle Notebook, or
     Option B: download locally with the Kaggle API).
  2. Set DATA_PATH below to point at the CSV file(s) you chose
     (e.g. one or more of 2009.csv ... 2018.csv, totaling ~4-7 GB).
  3. Run: python assignment_pipeline.py
  4. Outputs: sort_benchmark_results.csv, runtime_vs_n.png,
     insertion_orderings.png, and printed Part A/E numbers.

This script is written to run unchanged on the real ~4-7 GB dataset.
For demonstration/testing here (no internet access in this sandbox),
DEMO_MODE=True generates a small synthetic CSV with the SAME column
names/types as the real dataset, so every line of logic below is
exercised and verified before you point it at the real files.
"""

import os
import time
import random
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sort_algorithms import bubble_sort, selection_sort, insertion_sort

# ----------------------------------------------------------------------
# CONFIG — change these for the real run
# ----------------------------------------------------------------------
DEMO_MODE = True                       # set False once DATA_PATH is real Kaggle data
DATA_PATH = "./data/2018.csv"          # <- point this at your chosen file(s)
CHUNK_SIZE = 1_000_000
NUMERIC_COL = "ARR_DELAY"
TEXT_COL = "OP_CARRIER"

# Sample sizes to benchmark. For the REAL run, O(n^2) algorithms get slow
# fast: n=2000 -> ~4M comparisons, n=8000 -> ~64M. Start small, only go
# bigger if the previous size finished in a few seconds. 200-6000 is a
# reasonable range for bubble/selection/insertion on a laptop; don't run
# selection/bubble sort above ~10,000 unless you're prepared to wait minutes.
SAMPLE_SIZES = [200, 500, 1000, 2000, 4000]

RANDOM_SEED = 42


# ----------------------------------------------------------------------
# PART A — Data Loading & Exploration
# ----------------------------------------------------------------------
def load_data_chunked(path, chunk_size=CHUNK_SIZE):
    """
    Load a (possibly multi-GB) CSV in chunks rather than all at once.
    Returns: (total_row_count, column_list, one_chunk_memory_mb, full_df)

    NOTE: for the real multi-GB dataset you generally do NOT want to
    concatenate every chunk into one DataFrame (that defeats the purpose
    of chunked reading). Here we keep a running reservoir sample instead
    of the full data — see build_sample() below — which is the
    memory-safe way to get an n-row random sample out of a file too big
    to hold in RAM.
    """
    total_rows = 0
    columns = None
    first_chunk_mem_mb = None
    reservoir = None  # filled in by build_sample_from_chunks

    for i, chunk in enumerate(pd.read_csv(path, chunksize=chunk_size)):
        if columns is None:
            columns = list(chunk.columns)
            first_chunk_mem_mb = chunk.memory_usage(deep=True).sum() / 1e6
        total_rows += len(chunk)

    return total_rows, columns, first_chunk_mem_mb


def report_column_stats(df, numeric_col):
    """Part A.3 — basic stats for the chosen numeric column."""
    col = df[numeric_col]
    return {
        "min": col.min(),
        "max": col.max(),
        "mean": col.mean(),
        "pct_missing": 100 * col.isna().mean(),
    }


# ----------------------------------------------------------------------
# PART B — Building Samples
# ----------------------------------------------------------------------
def draw_random_sample(values, n, seed=RANDOM_SEED):
    """Draw a random sample of n values (Part B.1)."""
    rng = random.Random(seed)
    if n > len(values):
        raise ValueError(f"n={n} exceeds available rows={len(values)}")
    return rng.sample(list(values), n)


def make_ordering_variants(sample):
    """
    Part B.2 — three ordering variants representing:
      random   -> average case
      sorted   -> best case (already in order)
      reversed -> worst case
    """
    rnd = list(sample)
    srt = sorted(sample)
    rev = sorted(sample, reverse=True)
    return {"random": rnd, "sorted": srt, "reversed": rev}


# ----------------------------------------------------------------------
# PART D — Benchmark & Measure
# ----------------------------------------------------------------------
ALGORITHMS = {
    "bubble": bubble_sort,
    "selection": selection_sort,
    "insertion": insertion_sort,
}


def benchmark(values_by_size, column_label="numeric"):
    """
    Times every algorithm on every sample size and every ordering variant.
    Returns a list of result dicts -> becomes the results DataFrame/CSV.
    """
    rows = []
    for n, sample in values_by_size.items():
        variants = make_ordering_variants(sample)
        for ordering, data in variants.items():
            for algo_name, algo_fn in ALGORITHMS.items():
                arr = list(data)  # fresh copy each run
                start = time.perf_counter()
                algo_fn(arr)
                elapsed = time.perf_counter() - start
                assert arr == sorted(data), "sort correctness check failed"
                rows.append({
                    "size": n,
                    "ordering": ordering,
                    "algorithm": algo_name,
                    "column": column_label,
                    "time_seconds": elapsed,
                })
                print(f"  n={n:6d}  {ordering:8s}  {algo_name:10s}  "
                      f"{column_label:8s}  {elapsed:.6f}s")
    return rows


def plot_runtime_vs_n(results_df, out_path="runtime_vs_n.png"):
    """Part D.3 chart 1 — runtime vs n, one line per algorithm, random order."""
    subset = results_df[(results_df.ordering == "random") & (results_df.column == "numeric")]
    plt.figure(figsize=(7, 5))
    for algo in ALGORITHMS:
        d = subset[subset.algorithm == algo].sort_values("size")
        plt.plot(d["size"], d["time_seconds"], marker="o", label=algo)
    plt.xlabel("Sample size (n)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime vs n — random order, numeric column")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def plot_insertion_orderings(results_df, out_path="insertion_orderings.png"):
    """Part D.3 chart 2 — insertion sort: random vs sorted vs reversed."""
    subset = results_df[(results_df.algorithm == "insertion") & (results_df.column == "numeric")]
    plt.figure(figsize=(7, 5))
    for ordering in ["sorted", "random", "reversed"]:
        d = subset[subset.ordering == ordering].sort_values("size")
        plt.plot(d["size"], d["time_seconds"], marker="o", label=ordering)
    plt.xlabel("Sample size (n)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Insertion sort: best (sorted) vs average (random) vs worst (reversed)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


# ----------------------------------------------------------------------
# PART E — Extrapolate to Full Scale
# ----------------------------------------------------------------------
def estimate_c(results_df, algorithm, ordering="random", column="numeric"):
    """
    Fit time = c * n^2 using the LARGEST measured sample for this
    algorithm/ordering (Part E.1). c = time / n^2.
    """
    subset = results_df[
        (results_df.algorithm == algorithm)
        & (results_df.ordering == ordering)
        & (results_df.column == column)
    ].sort_values("size")
    largest = subset.iloc[-1]
    c = largest["time_seconds"] / (largest["size"] ** 2)
    return c, largest["size"], largest["time_seconds"]


def human_readable_seconds(seconds):
    """Convert seconds to the most readable unit (Part E.2)."""
    minute, hour, day, year = 60, 3600, 86400, 86400 * 365.25
    if seconds < minute:
        return f"{seconds:.2f} seconds"
    if seconds < hour:
        return f"{seconds/minute:.2f} minutes"
    if seconds < day:
        return f"{seconds/hour:.2f} hours"
    if seconds < year:
        return f"{seconds/day:.2f} days"
    return f"{seconds/year:.2f} years"


def extrapolate_full_scale(results_df, full_n, column="numeric"):
    """
    For each O(n^2) algorithm, estimate time to sort `full_n` rows,
    using c fit from the largest benchmarked sample. Also estimates
    Timsort (O(n log n), Python's built-in sorted()) for comparison
    and the resulting speedup factor.
    """
    print(f"\n--- Part E: extrapolation to full_n = {full_n:,} rows ---")
    report = {}
    for algo in ALGORITHMS:
        c, n0, t0 = estimate_c(results_df, algo, "random", column)
        est_seconds = c * (full_n ** 2)
        report[algo] = est_seconds
        print(f"{algo:10s}: c={c:.3e} (fit from n={n0}, t={t0:.6f}s) "
              f"-> estimated {human_readable_seconds(est_seconds)}")

    # Timsort comparison: measure actual sorted() time on the largest
    # sample and scale by n*log2(n) growth.
    largest_n = results_df["size"].max()
    largest_random = None
    for n, sample in SAMPLE_VALUES_CACHE.items():
        if n == largest_n:
            largest_random = sample
            break
    import math
    start = time.perf_counter()
    sorted(list(largest_random))
    t0 = time.perf_counter() - start
    c_timsort = t0 / (largest_n * math.log2(max(largest_n, 2)))
    est_timsort = c_timsort * full_n * math.log2(full_n)
    print(f"{'timsort':10s}: estimated {human_readable_seconds(est_timsort)}  "
          f"(Python's built-in sorted(), O(n log n))")

    print("\nSpeedup factor (O(n^2) time / Timsort time):")
    for algo, est_seconds in report.items():
        print(f"  {algo:10s}: {est_seconds / est_timsort:,.0f}x slower than Timsort")

    return report, est_timsort


# ----------------------------------------------------------------------
# DEMO DATA GENERATOR (only used when DEMO_MODE=True — replace with the
# real Kaggle CSV by setting DEMO_MODE=False and DATA_PATH correctly)
# ----------------------------------------------------------------------
def _make_demo_csv(path, n_rows=50_000, seed=RANDOM_SEED):
    rng = random.Random(seed)
    carriers = ["AA", "DL", "UA", "WN", "AS", "B6", "NK", "F9"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("FL_DATE,OP_CARRIER,ORIGIN,DEST,ARR_DELAY,DEP_DELAY,DISTANCE\n")
        for _ in range(n_rows):
            f.write(
                f"2018-01-01,{rng.choice(carriers)},ORD,LAX,"
                f"{rng.randint(-30, 200)},{rng.randint(-30, 200)},{rng.randint(100, 3000)}\n"
            )


SAMPLE_VALUES_CACHE = {}

if __name__ == "__main__":
    if DEMO_MODE:
        DATA_PATH = "./demo_data/demo_flights.csv"
        print(f"[DEMO_MODE] generating a small synthetic CSV at {DATA_PATH} "
              f"with the same columns as the real dataset, to test the pipeline.\n")
        _make_demo_csv(DATA_PATH, n_rows=50_000)

    # ---------------- Part A ----------------
    print("=== Part A: Data Loading & Exploration ===")
    total_rows, columns, chunk_mem_mb = load_data_chunked(DATA_PATH)
    print(f"Total rows loaded (chunked): {total_rows:,}")
    print(f"Columns: {columns}")
    print(f"Memory footprint of one chunk: {chunk_mem_mb:.2f} MB")

    # load full column data needed for sampling (numeric + text col only,
    # to keep memory reasonable even on the real multi-GB file)
    usecols = [NUMERIC_COL, TEXT_COL]
    df = pd.read_csv(DATA_PATH, usecols=usecols)
    stats = report_column_stats(df, NUMERIC_COL)
    print(f"\n{NUMERIC_COL} stats: min={stats['min']}, max={stats['max']}, "
          f"mean={stats['mean']:.2f}, missing={stats['pct_missing']:.2f}%")

    # ---------------- Part B ----------------
    print("\n=== Part B: Building Samples ===")
    numeric_values = df[NUMERIC_COL].dropna().tolist()
    text_values = df[TEXT_COL].dropna().tolist()

    values_by_size = {}
    for n in SAMPLE_SIZES:
        sample = draw_random_sample(numeric_values, n)
        values_by_size[n] = sample
        SAMPLE_VALUES_CACHE[n] = sample
    print(f"Built samples for sizes: {SAMPLE_SIZES}")

    # ---------------- Part D ----------------
    print("\n=== Part D: Benchmark & Measure (numeric column) ===")
    results = benchmark(values_by_size, column_label="numeric")

    # Part D.4 — repeat one sample size on the text column
    text_n = SAMPLE_SIZES[len(SAMPLE_SIZES) // 2]  # middle size
    text_sample = draw_random_sample(text_values, text_n)
    print(f"\n=== Part D.4: Benchmark on text column (n={text_n}) ===")
    results += benchmark({text_n: text_sample}, column_label="text")

    results_df = pd.DataFrame(results)
    results_df.to_csv("sort_benchmark_results.csv", index=False)
    print(f"\nSaved sort_benchmark_results.csv ({len(results_df)} rows)")

    plot_runtime_vs_n(results_df)
    plot_insertion_orderings(results_df)
    print("Saved runtime_vs_n.png and insertion_orderings.png")

    # ---------------- Part E ----------------
    # In the REAL run, set full_n to the actual row count of the dataset
    # portion you chose in Part A (e.g. total_rows from above).
    extrapolate_full_scale(results_df, full_n=total_rows, column="numeric")
