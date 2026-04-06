"""
Assignment 6 - Part 1: Selection Algorithms
Course: Introduction to Algorithms (CLRS 4th Edition)

Implements:
  - Deterministic selection in worst-case O(n) time (Median of Medians)
  - Randomized selection in expected O(n) time (Randomized Quickselect)

References:
  - CLRS Chapter 9: Medians and Order Statistics
"""

import random
import time
import sys


# ──────────────────────────────────────────────
# 1.  DETERMINISTIC SELECTION  (Median of Medians)
# ──────────────────────────────────────────────

def insertion_sort(arr: list, left: int, right: int) -> None:
    """
    In-place insertion sort of arr[left..right] (inclusive).
    Used internally to sort small groups of 5.
    Time: O(k^2) where k = right - left + 1 (k ≤ 5, so O(1) per call).
    """
    for i in range(left + 1, right + 1):
        key = arr[i]
        j = i - 1
        while j >= left and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


def partition_around(arr: list, left: int, right: int, pivot_value) -> tuple:
    """
    Three-way partition of arr[left..right] around pivot_value.
    Returns (low, high) such that:
      arr[left..low-1]  < pivot_value
      arr[low..high]   == pivot_value
      arr[high+1..right] > pivot_value

    Handles duplicates correctly, which is important for arrays
    with repeated elements.
    """
    # Move pivot to the right first
    for i in range(left, right + 1):
        if arr[i] == pivot_value:
            arr[i], arr[right] = arr[right], arr[i]
            break

    pivot = arr[right]
    store = left       # boundary for elements < pivot

    for i in range(left, right):
        if arr[i] < pivot:
            arr[i], arr[store] = arr[store], arr[i]
            store += 1

    arr[store], arr[right] = arr[right], arr[store]
    # Now arr[store] == pivot.  Expand equal region leftward.
    low = high = store

    # Merge any equal elements that ended up on the left
    i = low - 1
    while i >= left and arr[i] == pivot:
        arr[i], arr[low - 1] = arr[low - 1], arr[i]
        low -= 1
        i -= 1

    return low, high


def median_of_medians(arr: list, left: int, right: int, k: int) -> int:
    """
    Deterministic O(n) selection algorithm (CLRS §9.3).

    Returns the k-th smallest element (0-indexed) in arr[left..right].

    Strategy
    --------
    1. Divide arr into groups of 5.
    2. Sort each group and find its median.
    3. Recursively find the median of those medians → use as pivot.
    4. Partition arr around the pivot.
    5. Recurse into whichever side contains rank k.

    Correctness guarantee: the pivot is at least the median of ≈n/5
    groups, so at least 3·⌈n/10⌉ elements are both ≤ and ≥ the pivot.
    This ensures each recursive call reduces the problem size by a
    constant fraction, giving T(n) = T(7n/10 + 6) + T(n/5) + O(n) = O(n).
    """
    n = right - left + 1

    # Base case: tiny sub-array → sort and index directly
    if n <= 5:
        insertion_sort(arr, left, right)
        return arr[left + k]

    # ── Step 1 & 2: collect medians of each group of 5 ──
    medians = []
    i = left
    while i <= right:
        group_right = min(i + 4, right)
        insertion_sort(arr, i, group_right)
        mid = (i + group_right) // 2
        medians.append(arr[mid])
        i += 5

    # ── Step 3: find the median of medians recursively ──
    pivot_value = median_of_medians(
        medians, 0, len(medians) - 1, (len(medians) - 1) // 2
    )

    # ── Step 4: partition arr[left..right] around pivot_value ──
    low, high = partition_around(arr, left, right, pivot_value)

    # ── Step 5: recurse into the correct partition ──
    low_rank = low - left   # rank of first element equal to pivot
    high_rank = high - left # rank of last  element equal to pivot

    if k < low_rank:
        return median_of_medians(arr, left, low - 1, k)
    elif k > high_rank:
        return median_of_medians(arr, high + 1, right, k - high_rank - 1)
    else:
        return pivot_value


def deterministic_select(arr: list, k: int) -> int:
    """
    Public interface: return the k-th smallest element (1-indexed) of arr.
    Modifies a working copy; original array is unchanged.

    Parameters
    ----------
    arr : list of comparable elements
    k   : 1-based rank (1 = minimum, len(arr) = maximum)

    Returns
    -------
    The k-th smallest element.
    """
    if not arr:
        raise ValueError("Array must be non-empty.")
    if k < 1 or k > len(arr):
        raise IndexError(f"k={k} is out of range [1, {len(arr)}].")
    work = arr[:]                          # non-destructive copy
    return median_of_medians(work, 0, len(work) - 1, k - 1)


# ──────────────────────────────────────────────
# 2.  RANDOMIZED SELECTION  (Quickselect)
# ──────────────────────────────────────────────

def randomized_partition(arr: list, left: int, right: int) -> int:
    """
    Lomuto-style partition with a random pivot (CLRS §7.3).
    Picks a random index in [left, right], swaps it to the right,
    then partitions so that arr[p] is the final position of the pivot.

    Returns the index p of the placed pivot.
    """
    pivot_idx = random.randint(left, right)
    arr[pivot_idx], arr[right] = arr[right], arr[pivot_idx]

    pivot = arr[right]
    store = left
    for i in range(left, right):
        if arr[i] <= pivot:
            arr[i], arr[store] = arr[store], arr[i]
            store += 1
    arr[store], arr[right] = arr[right], arr[store]
    return store


def randomized_quickselect(arr: list, left: int, right: int, k: int) -> int:
    """
    Randomized selection algorithm (CLRS §9.2).

    Returns the k-th smallest element (0-indexed) in arr[left..right].

    Expected time O(n) because the random pivot gives a balanced
    partition on average; the probability of repeatedly hitting an
    adversarial split is exponentially small.
    """
    if left == right:
        return arr[left]

    pivot_pos = randomized_partition(arr, left, right)
    rank = pivot_pos - left  # 0-based rank of the pivot within this slice

    if k == rank:
        return arr[pivot_pos]
    elif k < rank:
        return randomized_quickselect(arr, left, pivot_pos - 1, k)
    else:
        return randomized_quickselect(arr, pivot_pos + 1, right, k - rank - 1)


def randomized_select(arr: list, k: int) -> int:
    """
    Public interface: return the k-th smallest element (1-indexed) of arr.
    Modifies a working copy; original array is unchanged.

    Parameters
    ----------
    arr : list of comparable elements
    k   : 1-based rank

    Returns
    -------
    The k-th smallest element.
    """
    if not arr:
        raise ValueError("Array must be non-empty.")
    if k < 1 or k > len(arr):
        raise IndexError(f"k={k} is out of range [1, {len(arr)}].")
    work = arr[:]
    return randomized_quickselect(work, 0, len(work) - 1, k - 1)


# ──────────────────────────────────────────────
# 3.  EMPIRICAL COMPARISON
# ──────────────────────────────────────────────

def benchmark(func, arr: list, k: int, repeats: int = 5) -> float:
    """
    Return the average elapsed seconds over `repeats` runs of func(arr, k).
    Each run gets a fresh copy of arr so no side-effects accumulate.
    """
    total = 0.0
    for _ in range(repeats):
        data = arr[:]
        start = time.perf_counter()
        func(data, k)
        total += time.perf_counter() - start
    return total / repeats


def run_empirical_analysis():
    """
    Benchmark both selection algorithms on three input distributions
    and a range of sizes.  Print a formatted table.
    """
    sizes = [1_000, 5_000, 10_000, 50_000, 100_000]
    distributions = {
        "Random":         lambda n: [random.randint(0, n) for _ in range(n)],
        "Sorted":         lambda n: list(range(n)),
        "Reverse-sorted": lambda n: list(range(n, 0, -1)),
    }

    header = f"{'Distribution':<17} {'n':>8}  {'Det (ms)':>10}  {'Rand (ms)':>10}"
    print("\n" + "=" * len(header))
    print("Empirical Performance Comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for dist_name, dist_fn in distributions.items():
        for n in sizes:
            arr = dist_fn(n)
            k   = n // 2           # always find the median

            t_det  = benchmark(deterministic_select, arr, k) * 1000
            t_rand = benchmark(randomized_select,    arr, k) * 1000

            print(f"{dist_name:<17} {n:>8}  {t_det:>10.3f}  {t_rand:>10.3f}")
        print()


# ──────────────────────────────────────────────
# 4.  QUICK CORRECTNESS SMOKE-TEST
# ──────────────────────────────────────────────

def verify_correctness():
    """Compare both algorithms against Python's built-in sorted()."""
    print("Correctness verification:")
    test_cases = [
        [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5],
        [7],
        [2, 2, 2, 2],
        list(range(20, 0, -1)),
        [random.randint(-100, 100) for _ in range(200)],
    ]
    all_pass = True
    for arr in test_cases:
        expected = sorted(arr)
        for k in range(1, len(arr) + 1):
            det  = deterministic_select(arr, k)
            rand = randomized_select(arr, k)
            if det != expected[k - 1] or rand != expected[k - 1]:
                print(f"  FAIL  k={k}  expected={expected[k-1]}  det={det}  rand={rand}")
                all_pass = False
    print("  All tests passed!\n" if all_pass else "  Some tests FAILED.\n")


# ──────────────────────────────────────────────
# 5.  ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    sys.setrecursionlimit(200_000)
    verify_correctness()
    run_empirical_analysis()
