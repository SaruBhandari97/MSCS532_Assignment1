"""
quicksort.py
------------
Deterministic and Randomized Quicksort implementations with
empirical benchmarking across different input distributions.

Course : MSCS-532-B01 – Algorithms and Data Structures (Spring 2026)
Author : Saru Bhandari
Text   : Cormen et al. (2022). Introduction to Algorithms (4th ed.). MIT Press.
"""

import random
import time
import sys

# Increase recursion limit for large sorted/reverse-sorted inputs
sys.setrecursionlimit(20000)


# ─────────────────────────────────────────────
# 1. Deterministic Quicksort  (last element as pivot)
# ─────────────────────────────────────────────

def partition(arr, low, high):
    """
    Partition arr[low..high] around the last element as pivot.

    All elements <= pivot end up to its left; all elements > pivot
    end up to its right.  Returns the final index of the pivot.

    Time complexity : O(n) where n = high - low + 1
    """
    pivot = arr[high]
    i = low - 1                        # index of the smaller element

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    # Place pivot in its correct position
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def quicksort(arr, low, high):
    """
    Deterministic Quicksort using last-element pivot selection.

    Parameters
    ----------
    arr  : list  – the array to sort (sorted in place)
    low  : int   – left boundary index
    high : int   – right boundary index

    Time complexity:
        Best / Average : O(n log n)
        Worst          : O(n²)  — occurs on already-sorted input
    Space complexity   : O(log n) average (call stack)
    """
    if low < high:
        pivot_index = partition(arr, low, high)
        quicksort(arr, low, pivot_index - 1)
        quicksort(arr, pivot_index + 1, high)


# ─────────────────────────────────────────────
# 2. Randomized Quicksort  (random pivot)
# ─────────────────────────────────────────────

def randomized_partition(arr, low, high):
    """
    Choose a random pivot from arr[low..high], swap it to the end,
    then delegate to the standard partition function.

    Randomization breaks the adversarial input structure that causes
    O(n²) behaviour in the deterministic version.

    Time complexity : O(n)
    """
    rand_index = random.randint(low, high)
    arr[rand_index], arr[high] = arr[high], arr[rand_index]
    return partition(arr, low, high)


def randomized_quicksort(arr, low, high):
    """
    Randomized Quicksort — pivot chosen uniformly at random.

    Parameters
    ----------
    arr  : list  – the array to sort (sorted in place)
    low  : int   – left boundary index
    high : int   – right boundary index

    Time complexity:
        Expected (all inputs) : O(n log n)
        Worst case            : O(n²)  — with negligible probability
    Space complexity          : O(log n) expected (call stack)
    """
    if low < high:
        pivot_index = randomized_partition(arr, low, high)
        randomized_quicksort(arr, low, pivot_index - 1)
        randomized_quicksort(arr, pivot_index + 1, high)


# ─────────────────────────────────────────────
# 3. Empirical Benchmarking
# ─────────────────────────────────────────────

def time_sort(sort_fn, arr):
    """Run sort_fn on a copy of arr and return elapsed time in seconds."""
    data = arr[:]
    start = time.perf_counter()
    sort_fn(data, 0, len(data) - 1)
    return time.perf_counter() - start


def benchmark():
    """
    Compare deterministic vs randomized Quicksort across three input
    distributions (random, sorted, reverse-sorted) and three sizes.
    """
    sizes = [500, 1000, 2000]
    distributions = {
        "Random"        : lambda n: [random.randint(0, 10_000) for _ in range(n)],
        "Sorted"        : lambda n: list(range(n)),
        "Reverse-sorted": lambda n: list(range(n, 0, -1)),
    }

    header = f"{'Distribution':<18} {'Size':>6}  {'Deterministic (s)':>18}  {'Randomized (s)':>15}"
    print(header)
    print("-" * len(header))

    for dist_name, gen in distributions.items():
        for n in sizes:
            arr = gen(n)
            t_det  = time_sort(quicksort, arr)
            t_rand = time_sort(randomized_quicksort, arr)
            print(f"{dist_name:<18} {n:>6}  {t_det:>18.6f}  {t_rand:>15.6f}")
        print()


# ─────────────────────────────────────────────
# 4. Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Correctness check
    sample = [3, 6, 8, 10, 1, 2, 1]
    det  = sample[:]
    rand = sample[:]

    quicksort(det, 0, len(det) - 1)
    randomized_quicksort(rand, 0, len(rand) - 1)

    print("Original :", sample)
    print("Det sort :", det)
    print("Rand sort:", rand)
    print()

    # Benchmarks
    benchmark()
