"""
plot_results.py
===============
Creates visual charts for:
1. Quicksort performance comparison
2. Hash table performance analysis

Make sure to run the sorting and hash table scripts first,
so the data is ready for plotting.

Requirement:
    matplotlib (install using: pip install matplotlib)
"""

import random
import sys
import time
import matplotlib
matplotlib.use("Agg")          # Use a backend that works without a GUI
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Import functions/classes from your own modules (same folder required)
from randomized_quicksort import (
    randomized_quicksort, deterministic_quicksort,
    generate_input, time_sort
)
from hash_table import HashTableChaining, benchmark_operations

# Increase recursion limit to prevent crashes in worst-case deterministic quicksort
sys.setrecursionlimit(100_000)


# ============================================================
# Global styling for plots
# ============================================================

COLORS = {
    "randomized":    "#2196F3",   # Blue for randomized quicksort
    "deterministic": "#F44336",   # Red for deterministic quicksort
    "insert":        "#4CAF50",   # Green for insert operations
    "search":        "#2196F3",   # Blue for search operations
    "delete":        "#FF9800",   # Orange for delete operations
    "load":          "#9C27B0",   # Purple for load factor
}

# Apply consistent styling across all plots
plt.rcParams.update({
    "font.family":  "monospace",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        120,
})


# ============================================================
# Part 1 — Quicksort benchmarking
# ============================================================

def collect_quicksort_data(sizes, input_kinds, trials=3):
    """
    Runs both randomized and deterministic quicksort
    and collects average execution times.

    Output format:
        data[kind] = {
            "sizes": [...],
            "rand":  [...],
            "det":   [...]
        }
    """
    data = {kind: {"sizes": [], "rand": [], "det": []} for kind in input_kinds}

    for kind in input_kinds:
        print(f"  Running quicksort benchmarks for input type: {kind}")
        for size in sizes:
            r_times, d_times = [], []

            for _ in range(trials):
                arr = generate_input(kind, size)

                # Measure randomized quicksort time
                r_times.append(time_sort(randomized_quicksort, arr))

                # Measure deterministic quicksort (may fail for bad inputs)
                try:
                    d_times.append(time_sort(deterministic_quicksort, arr))
                except RecursionError:
                    d_times.append(None)

            # Filter out failed runs
            valid_r = [t for t in r_times if t is not None]
            valid_d = [t for t in d_times if t is not None]

            # Store average times
            data[kind]["sizes"].append(size)
            data[kind]["rand"].append(
                sum(valid_r) / len(valid_r) if valid_r else None)
            data[kind]["det"].append(
                sum(valid_d) / len(valid_d) if valid_d else None)

    return data


def plot_quicksort(data, input_kinds, output_path="quicksort_comparison.png"):
    """
    Creates a 2x2 grid of plots, one for each input type.
    Each plot compares randomized vs deterministic quicksort.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    fig.suptitle(
        "Randomized vs Deterministic Quicksort\nAverage Execution Time",
        fontsize=14, fontweight="bold", y=1.01
    )

    panel_titles = {
        "random":   "Random Input",
        "sorted":   "Already Sorted Input",
        "reverse":  "Reverse Sorted Input",
        "repeated": "Repeated Elements",
    }

    for ax, kind in zip(axes.flat, input_kinds):
        sizes = data[kind]["sizes"]
        rand  = data[kind]["rand"]
        det   = data[kind]["det"]

        # Plot randomized quicksort (always works)
        ax.plot(
            sizes, rand,
            color=COLORS["randomized"],
            linewidth=2,
            marker="o",
            markersize=4,
            label="Randomized QS"
        )

        # Plot deterministic quicksort (skip failed runs)
        det_pairs = [(s, t) for s, t in zip(sizes, det) if t is not None]
        if det_pairs:
            xs, ys = zip(*det_pairs)
            ax.plot(
                xs, ys,
                color=COLORS["deterministic"],
                linewidth=2,
                marker="s",
                markersize=4,
                linestyle="--",
                label="Deterministic QS"
            )

        # Display warning if recursion limit was hit
        if any(t is None for t in det):
            ax.text(
                0.97, 0.05,
                "⚠ Deterministic QS exceeded recursion limit\nfor large inputs",
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=7,
                color=COLORS["deterministic"],
                bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3E0", ec="#FF9800")
            )

        ax.set_title(panel_titles[kind], fontsize=11, fontweight="bold")
        ax.set_xlabel("Input size (n)")
        ax.set_ylabel("Execution time (seconds)")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)

        # Format x-axis with commas for readability
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
        )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"  Chart saved: {output_path}")
    plt.close()


# ============================================================
# Part 2 — Hash Table benchmarking
# ============================================================

def collect_hash_data(sizes):
    """
    Runs performance tests for the hash table
    and collects timing + load factor information.
    """
    rows = []
    for n in sizes:
        print(f"  Running hash table benchmark for n={n}")
        r = benchmark_operations(n)
        rows.append(r)
    return rows


def plot_hash_table(rows, output_path="hashtable_performance.png"):
    """
    Creates two plots:
    1. Time taken for insert, search, delete operations
    2. Load factor and maximum chain length
    """
    ns       = [r["n"] for r in rows]
    inserts  = [r["insert_time"] for r in rows]
    searches = [r["search_time"] for r in rows]
    deletes  = [r["delete_time"] for r in rows]
    loads    = [r["load_after_insert"] for r in rows]
    chains   = [r["max_chain_length"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    fig.suptitle(
        "Hash Table (Chaining) — Performance Overview",
        fontsize=14, fontweight="bold"
    )

    # --- Plot 1: Operation timing ---
    ax1.plot(ns, inserts,  color=COLORS["insert"], marker="o", linewidth=2, label="Insert")
    ax1.plot(ns, searches, color=COLORS["search"], marker="s", linewidth=2, label="Search")
    ax1.plot(ns, deletes,  color=COLORS["delete"], marker="^", linewidth=2, label="Delete")

    ax1.set_title("Execution Time vs Number of Elements", fontweight="bold")
    ax1.set_xlabel("Number of elements (n)")
    ax1.set_ylabel("Time (seconds)")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax1.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    # --- Plot 2: Load factor + chain length ---
    color_load  = COLORS["load"]
    color_chain = "#795548"

    ax2.plot(ns, loads, color=color_load, marker="o", linewidth=2, label="Load factor (α)")
    ax2.axhline(0.75, color=color_load, linestyle="--", linewidth=1,
                alpha=0.6, label="Threshold (0.75)")

    ax2.set_xlabel("Number of elements (n)")
    ax2.set_ylabel("Load factor", color=color_load)
    ax2.tick_params(axis="y", labelcolor=color_load)
    ax2.set_ylim(0, 1.0)

    # Secondary axis for chain length
    ax3 = ax2.twinx()
    ax3.plot(ns, chains, color=color_chain, marker="s",
             linewidth=2, linestyle=":", label="Max chain length")

    ax3.set_ylabel("Max chain length", color=color_chain)
    ax3.tick_params(axis="y", labelcolor=color_chain)

    # Merge legends from both axes
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    ax2.set_title("Load Factor and Collision Behavior", fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.5)

    ax2.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    ax3.spines["top"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"  Chart saved: {output_path}")
    plt.close()


# ============================================================
# Chain length distribution visualization
# ============================================================

def plot_chain_distribution(n=10_000, output_path="chain_distribution.png"):
    """
    Shows how elements are distributed across chains
    (i.e., how collisions are spread in the hash table).
    """
    from collections import Counter

    ht = HashTableChaining()

    # Insert sequential keys
    for i in range(n):
        ht.insert(i, i)

    lengths = ht.chain_lengths()
    dist    = Counter(lengths)

    xs = sorted(dist.keys())
    ys = [dist[x] for x in xs]

    fig, ax = plt.subplots(figsize=(8, 4))

    bars = ax.bar(xs, ys, color=COLORS["insert"],
                  edgecolor="white", linewidth=0.5)

    # Highlight the tallest bar
    max_bar = max(ys)
    for bar, y in zip(bars, ys):
        if y == max_bar:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y + max_bar * 0.01,
                f"{y}",
                ha="center", va="bottom", fontsize=9
            )

    ax.set_title(
        f"Chain Length Distribution (n={n:,}, m={ht.num_slots:,}, α={ht.load_factor:.2f})",
        fontweight="bold"
    )
    ax.set_xlabel("Chain length")
    ax.set_ylabel("Number of slots")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"  Chart saved: {output_path}")
    plt.close()


# ============================================================
# Main execution
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  Generating Quicksort charts...")
    print("=" * 55)

    QS_SIZES = [100, 500, 1000, 2500, 5000, 10000]
    INPUT_KINDS = ["random", "sorted", "reverse", "repeated"]

    qs_data = collect_quicksort_data(QS_SIZES, INPUT_KINDS, trials=3)
    plot_quicksort(qs_data, INPUT_KINDS)

    print("\n" + "=" * 55)
    print("  Generating Hash Table charts...")
    print("=" * 55)

    HT_SIZES = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000]

    ht_data = collect_hash_data(HT_SIZES)
    plot_hash_table(ht_data)
    plot_chain_distribution(n=10_000)

    print("\nAll charts generated successfully")