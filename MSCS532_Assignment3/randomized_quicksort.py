"""
randomized_quicksort.py
=======================
This module implements two versions of Quicksort:
1. Randomized Quicksort (robust against bad input cases)
2. Deterministic Quicksort (fixed pivot — used for comparison)

It also includes helper functions to:
- Generate different types of input data
- Measure execution time
- Run performance benchmarks

Assignment 3 — Part 1
"""

import random
import time
import sys

# Increase recursion depth to handle deep recursion cases
# (especially important for deterministic quicksort on sorted inputs)
sys.setrecursionlimit(100_000)


# ---------------------------------------------------------------------------
# Randomized Quicksort
# ---------------------------------------------------------------------------

def randomized_partition(arr, low, high):
    """
    Partition the array using a randomly selected pivot.

    Instead of always picking a fixed pivot (like first or last element),
    we randomly choose one element within the range. This helps avoid
    consistently poor splits, which can lead to worst-case performance.

    Steps:
    1. Pick a random pivot
    2. Move it to the end
    3. Apply standard partition logic (Lomuto scheme)

    Returns:
        The final index of the pivot after partitioning
    """
    # Select a random pivot index within the current range
    pivot_index = random.randint(low, high)

    # Move pivot to the end so we can reuse standard partition logic
    arr[pivot_index], arr[high] = arr[high], arr[pivot_index]

    pivot = arr[high]
    i = low - 1  # Marks boundary of elements smaller than pivot

    for j in range(low, high):
        # If current element should be on the left side
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    # Place pivot in its correct sorted position
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def randomized_quicksort(arr, low=None, high=None):
    """
    Sorts the array using randomized quicksort.

    Why randomized?
    - It reduces the chance of hitting worst-case O(n²)
    - Expected performance is O(n log n) regardless of input order

    The function works recursively:
    - Partition the array
    - Recursively sort left and right subarrays
    """
    # Initialize bounds on first call
    if low is None:
        low = 0
    if high is None:
        high = len(arr) - 1

    # Only proceed if there is more than one element
    if low < high:
        pivot_pos = randomized_partition(arr, low, high)

        # Recursively sort left and right partitions
        randomized_quicksort(arr, low, pivot_pos - 1)
        randomized_quicksort(arr, pivot_pos + 1, high)


# ---------------------------------------------------------------------------
# Deterministic Quicksort (fixed pivot)
# ---------------------------------------------------------------------------

def deterministic_partition(arr, low, high):
    """
    Partition using the first element as pivot.

    This is a simple and commonly taught approach, but it has a weakness:
    if the input is already sorted (or reverse sorted), the pivot choice
    becomes consistently poor.

    Result:
    - One side has size 0
    - Other side has size n-1
    → Leads to O(n²) time complexity
    """
    pivot = arr[low]

    # Move pivot to the end to reuse Lomuto partition logic
    arr[low], arr[high] = arr[high], arr[low]

    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def deterministic_quicksort(arr, low=None, high=None):
    """
    Sorts the array using a fixed pivot strategy.

    Key idea:
    - Always uses the first element as pivot

    Trade-off:
    - Works fine on random input
    - Performs very poorly on sorted or nearly sorted data

    This version is included to highlight why pivot selection matters.
    """
    if low is None:
        low = 0
    if high is None:
        high = len(arr) - 1

    if low < high:
        pivot_pos = deterministic_partition(arr, low, high)

        deterministic_quicksort(arr, low, pivot_pos - 1)
        deterministic_quicksort(arr, pivot_pos + 1, high)


# ---------------------------------------------------------------------------
# Input generation utilities
# ---------------------------------------------------------------------------

def generate_input(kind, size):
    """
    Generates input arrays of different characteristics.

    Types:
    - 'random'   → random values (general case)
    - 'sorted'   → already sorted (best/worst depending on algorithm)
    - 'reverse'  → descending order
    - 'repeated' → limited unique values (many duplicates)

    These variations help analyze algorithm behavior under different conditions.
    """
    if kind == "random":
        return [random.randint(0, size * 10) for _ in range(size)]
    elif kind == "sorted":
        return list(range(size))
    elif kind == "reverse":
        return list(range(size, 0, -1))
    elif kind == "repeated":
        return [random.randint(0, 9) for _ in range(size)]
    else:
        raise ValueError(f"Unknown input kind: {kind}")


def time_sort(sort_fn, data):
    """
    Measures execution time of a sorting function.

    Important:
    - The input array is copied before sorting
    - This ensures each run starts with identical data

    Returns:
        Time taken (in seconds)
    """
    arr = data[:]  # Copy input to preserve original data
    start = time.perf_counter()
    sort_fn(arr)
    end = time.perf_counter()
    return end - start


def run_benchmark(sizes, input_kinds, trials=3):
    """
    Runs performance tests for both sorting algorithms.

    For each (input type, size):
    - Runs multiple trials
    - Averages results to reduce noise

    Handles edge case:
    - Deterministic quicksort may fail with RecursionError
      on large sorted inputs → recorded as None
    """
    results = {kind: {} for kind in input_kinds}

    for kind in input_kinds:
        for size in sizes:
            rand_times = []
            det_times  = []

            for _ in range(trials):
                data = generate_input(kind, size)

                rand_times.append(time_sort(randomized_quicksort, data))

                try:
                    det_times.append(time_sort(deterministic_quicksort, data))
                except RecursionError:
                    det_times.append(None)

            valid_rand = [t for t in rand_times if t is not None]
            valid_det  = [t for t in det_times if t is not None]

            results[kind][size] = {
                "randomized":    sum(valid_rand) / len(valid_rand) if valid_rand else None,
                "deterministic": sum(valid_det)  / len(valid_det)  if valid_det  else None,
            }

            # Print progress for visibility during long runs
            r_str = f"{results[kind][size]['randomized']:.4f}s" \
                    if results[kind][size]['randomized'] is not None else "FAILED"
            d_str = f"{results[kind][size]['deterministic']:.4f}s" \
                    if results[kind][size]['deterministic'] is not None else "FAILED (RecursionError)"

            print(f"  [{kind:>10s}] n={size:>6d} | Rand: {r_str:>12s}  Det: {d_str}")

    return results


# ---------------------------------------------------------------------------
# Correctness validation
# ---------------------------------------------------------------------------

def verify_sort(sort_fn, name):
    """
    Verifies correctness by comparing output with Python's built-in sort.

    Uses different edge cases:
    - Empty list
    - Single element
    - Sorted and reverse-sorted arrays
    - Duplicate values

    Helps confirm implementation is logically correct before benchmarking.
    """
    test_cases = [
        [],
        [42],
        [3, 1, 4, 1, 5, 9, 2, 6],
        list(range(20)),
        list(range(19, -1, -1)),
        [7] * 15,
    ]

    all_passed = True
    for tc in test_cases:
        arr = tc[:]
        sort_fn(arr)
        if arr != sorted(tc):
            print(f"  FAIL {name} on input {tc}")
            all_passed = False

    if all_passed:
        print(f"  PASS — {name} works correctly on all test cases")


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  CORRECTNESS VERIFICATION")
    print("=" * 60)

    verify_sort(randomized_quicksort,   "Randomized Quicksort")
    verify_sort(deterministic_quicksort, "Deterministic Quicksort")

    print()
    print("=" * 60)
    print("  PERFORMANCE BENCHMARK")
    print("=" * 60)
    print("(averaging 3 trials per configuration)\n")

    # Selected sizes balance runtime and meaningful comparison
    SIZES       = [100, 500, 1000, 2500, 5000, 10000]
    INPUT_KINDS = ["random", "sorted", "reverse", "repeated"]

    results = run_benchmark(SIZES, INPUT_KINDS, trials=3)

    # Display results in a readable table format
    print()
    print("=" * 60)
    print("  SUMMARY TABLE  (times in seconds, None = RecursionError)")
    print("=" * 60)

    for kind in INPUT_KINDS:
        print(f"\n  Input type: {kind.upper()}")
        print(f"  {'n':>8}  {'Randomized':>14}  {'Deterministic':>16}")
        print(f"  {'-'*8}  {'-'*14}  {'-'*16}")

        for size in SIZES:
            r = results[kind][size]["randomized"]
            d = results[kind][size]["deterministic"]

            r_str = f"{r:.6f}" if r is not None else "      None"
            d_str = f"{d:.6f}" if d is not None else "          None"

            print(f"  {size:>8}  {r_str:>14}  {d_str:>16}")