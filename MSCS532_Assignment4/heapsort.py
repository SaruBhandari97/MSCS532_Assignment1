"""
heapsort.py
===========
Assignment 4 – Heap Data Structures: Implementation, Analysis, and Applications
Course:  Data Structures and Algorithms

Overview
--------
This module implements the Heapsort algorithm from scratch using an array-based
binary max-heap.  It also implements Quicksort and Merge Sort as comparison
baselines, and provides a benchmarking harness that measures wall-clock
performance across multiple input sizes and distributions.

Design Decisions
----------------
1. Array-based heap (not a linked tree)
   The parent-child relationship is encoded arithmetically:
       parent(i)      = (i - 1) // 2
       left_child(i)  = 2 * i + 1
       right_child(i) = 2 * i + 2
   This eliminates pointer overhead, keeps memory contiguous, and makes
   index arithmetic the only cost of traversal.

2. Max-heap for descending extraction
   After build_max_heap the largest element sits at index 0.  We repeatedly
   swap it to the end of the "active" portion of the array and shrink the
   heap boundary, producing a sorted array in ascending order without
   needing a second pass or extra memory.

3. Heapsort operates on a copy
   The public heapsort() function works on arr[:] so the caller's list is
   never mutated.  Internal helpers (heapify, build_max_heap) mutate
   in-place for efficiency.

4. Median-of-three pivot for Quicksort
   Naive Quicksort degrades to O(n^2) on sorted inputs.  Choosing the median
   of arr[low], arr[mid], arr[high] as the pivot dramatically reduces the
   probability of degenerate partitions, making the comparison fair.

Complexity Summary
------------------
Algorithm     Best        Average     Worst       Space
----------    ----------  ----------  ----------  -------
Heapsort      O(n log n)  O(n log n)  O(n log n)  O(log n)*
Quicksort     O(n log n)  O(n log n)  O(n^2)      O(log n)
Merge Sort    O(n log n)  O(n log n)  O(n log n)  O(n)

* Recursive heapify uses O(log n) call-stack depth.  An iterative
  implementation achieves O(1) auxiliary space.

Usage
-----
    from heapsort import heapsort
    sorted_list = heapsort([5, 3, 8, 1, 9, 2])

    # Run full benchmark suite and return raw timing data
    from heapsort import run_benchmarks
    sizes, results = run_benchmarks()
"""

import time
import random
import sys


# ==============================================================================
# SECTION 1 -- Core Heap Operations
# ==============================================================================

def heapify(arr: list, n: int, i: int) -> None:
    """
    Restore the max-heap property for the subtree rooted at index ``i``.

    This is the fundamental "sift-down" operation.  It assumes that the
    left and right subtrees of node i are already valid max-heaps.  The
    function compares node i with its children and, if a child is larger,
    swaps them and recurses on the affected subtree.

    Why start from i and go DOWN (not up)?
    ----------------------------------------
    build_max_heap calls heapify on every non-leaf starting from the
    bottom of the tree.  At each step, both subtrees are already heapified,
    so only the root of the current subtree may be out of place -- we fix
    it by pushing it downward until it lands in a valid position.

    Parameters
    ----------
    arr : list
        The array that stores the heap.  Modified in-place.
    n   : int
        The current heap size (may be less than len(arr) during extraction).
    i   : int
        Index of the root of the subtree to heapify.

    Time Complexity
    ---------------
    O(log n) -- the element travels at most one root-to-leaf path, whose
    length is floor(log2 n).
    """
    largest = i            # Tentatively, assume the root is the largest node
    left    = 2 * i + 1   # Left child lives at this index (array formula)
    right   = 2 * i + 2   # Right child lives at this index (array formula)

    # -- Compare root with left child ----------------------------------------
    # We must check left < n first to avoid an IndexError on a leaf node.
    if left < n and arr[left] > arr[largest]:
        largest = left      # Left child beats the current candidate

    # -- Compare current largest with right child ----------------------------
    if right < n and arr[right] > arr[largest]:
        largest = right     # Right child is the new candidate

    # -- If root is NOT the largest, fix the violation -----------------------
    if largest != i:
        # Swap the root with the larger child to restore local heap order
        arr[i], arr[largest] = arr[largest], arr[i]

        # The swap may have broken the heap property in the subtree rooted
        # at 'largest', so we recurse there to propagate the fix downward.
        heapify(arr, n, largest)


def build_max_heap(arr: list) -> None:
    """
    Re-arrange an arbitrary array into a valid max-heap in-place.

    Strategy -- Bottom-up heapification
    -------------------------------------
    Leaf nodes (indices n//2 through n-1) trivially satisfy the heap
    property because they have no children.  We therefore start at the last
    *non-leaf* node (index n//2 - 1) and call heapify on every node up to
    the root (index 0).  This guarantees that when heapify is called on
    node i, both of its subtrees are already valid heaps.

    Why is this O(n) and not O(n log n)?
    --------------------------------------
    Nodes near the bottom of the tree have small subtree heights, so
    heapify is cheap for most of them.  Summing the work over all levels
    using the identity sum(k / 2^k) = 2 yields a total cost of O(n).
    This is faster than inserting elements one-by-one (which is O(n log n)).

    Parameters
    ----------
    arr : list
        Array to convert.  Modified in-place.

    Time Complexity
    ---------------
    O(n) amortised -- linear despite n/2 calls to O(log n) heapify.
    """
    n = len(arr)
    # Iterate backward from the last internal node to the root.
    # Index n//2 - 1 is the parent of the last element (index n-1).
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)   # Each call costs O(height of subtree at i)


def heapsort(arr: list) -> list:
    """
    Sort a list in ascending order using the Heapsort algorithm.

    Algorithm Walk-Through
    ----------------------
    Phase 1 -- Build max-heap  [O(n)]
        Rearrange the array so that arr[0] is the global maximum and every
        parent is >= its children.

    Phase 2 -- Sorted extraction  [O(n log n)]
        For i = n-1 down to 1:
            1. Swap arr[0] (current max) with arr[i].
               -> The maximum is now in its final sorted position at arr[i].
            2. Conceptually shrink the heap to size i (ignore arr[i..n-1]).
            3. Call heapify(arr, i, 0) to restore the heap property over
               the reduced heap arr[0..i-1].
        After n-1 iterations the entire array is sorted ascending.

    Why O(n log n) in ALL cases?
    -----------------------------
    Unlike Quicksort, Heapsort has no "lucky" or "unlucky" input.  Every
    extraction always costs exactly O(log i) regardless of the values
    involved, because heap height is determined only by size, not content.

    Parameters
    ----------
    arr : list
        Input list.  The original is NOT modified (a copy is sorted).

    Returns
    -------
    list
        A new sorted list in ascending order.

    Time Complexity
    ---------------
    O(n log n) -- worst, average, and best cases.

    Space Complexity
    ----------------
    O(log n) auxiliary -- dominated by the recursive call stack of heapify.
    The sort itself is in-place on the working copy.
    """
    result = arr[:]      # Shallow copy -- we never mutate the caller's list
    n = len(result)

    # -- Phase 1: Transform the array into a max-heap ------------------------
    build_max_heap(result)
    # At this point result[0] holds the largest element.

    # -- Phase 2: Extract elements from the heap one by one -----------------
    # i tracks the boundary between the heap (result[0..i]) and the already-
    # sorted suffix (result[i+1..n-1]).
    for i in range(n - 1, 0, -1):
        # Step 2a: The root (largest remaining element) belongs at position i
        result[0], result[i] = result[i], result[0]

        # Step 2b: The heap now covers result[0..i-1].  The new root may
        # violate the heap property, so restore it with heapify.
        # Passing i (not n) as heap size excludes the sorted suffix.
        heapify(result, i, 0)

    return result   # result[0] <= result[1] <= ... <= result[n-1]


# ==============================================================================
# SECTION 2 -- Comparison Sorting Algorithms (Benchmark Baselines)
# ==============================================================================

def quicksort(arr: list) -> list:
    """
    Sort a list using Quicksort with a median-of-three pivot strategy.

    Median-of-three selects the pivot as the median of arr[low], arr[mid],
    and arr[high].  This prevents O(n^2) degeneration on already-sorted or
    reverse-sorted inputs that would plague a naive "last element" pivot.

    Time Complexity
    ---------------
    Best / Average : O(n log n)
    Worst          : O(n^2) -- rare with median-of-three, but theoretically
                     possible on adversarially crafted inputs.

    Space Complexity : O(log n) call stack.
    """
    result = arr[:]                        # Work on a copy
    _quicksort(result, 0, len(result) - 1)
    return result


def _quicksort(arr: list, low: int, high: int) -> None:
    """
    Recursive in-place Quicksort over arr[low..high].

    Base case: a subarray of length 0 or 1 is already sorted, so we return
    immediately when low >= high.
    """
    if low < high:
        # Partition the subarray and get the final position of the pivot
        pi = _partition(arr, low, high)

        # Recursively sort the two partitions on either side of the pivot.
        # The pivot itself is already in its correct final sorted position.
        _quicksort(arr, low, pi - 1)
        _quicksort(arr, pi + 1, high)


def _partition(arr: list, low: int, high: int) -> int:
    """
    Partition arr[low..high] around a median-of-three pivot.

    Steps
    -----
    1. Sort arr[low], arr[mid], arr[high] in place so that
       arr[low] <= arr[mid] <= arr[high].
    2. Move the median (pivot) to arr[high] for the partitioning loop.
    3. Walk pointer i forward, swapping elements <= pivot to the left side.
    4. Place the pivot at its final index and return that index.

    Returns
    -------
    int
        The final sorted position of the pivot element.
    """
    mid = (low + high) // 2

    # -- Median-of-three: sort the three boundary elements in place ----------
    # After these three conditional swaps, arr[low] <= arr[mid] <= arr[high].
    if arr[low] > arr[mid]:
        arr[low], arr[mid] = arr[mid], arr[low]
    if arr[low] > arr[high]:
        arr[low], arr[high] = arr[high], arr[low]
    if arr[mid] > arr[high]:
        arr[mid], arr[high] = arr[high], arr[mid]

    # -- Place pivot (the median) at arr[high] for the partitioning loop -----
    pivot = arr[mid]
    arr[mid], arr[high] = arr[high], arr[mid]

    # -- Lomuto-style partition loop -----------------------------------------
    # i tracks the right boundary of elements known to be <= pivot.
    # j scans through the unsorted portion.
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]   # Pull small element to the left

    # Place pivot in its correct sorted position (between the two partitions)
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1    # Return the pivot's final index


def merge_sort(arr: list) -> list:
    """
    Sort a list using top-down recursive Merge Sort.

    Merge Sort is stable and guarantees O(n log n) in all cases, but
    requires O(n) auxiliary memory for the merge step -- unlike Heapsort's
    O(1) in-place operation.  It is included here as a reference baseline
    that shares Heapsort's asymptotic bound while differing in constant
    factors and memory behavior.

    Time Complexity  : O(n log n) -- all cases.
    Space Complexity : O(n) -- the merge step allocates a new list each call.
    """
    # Base case: a list of 0 or 1 elements is already sorted by definition
    if len(arr) <= 1:
        return arr[:]

    # -- Divide --------------------------------------------------------------
    mid   = len(arr) // 2
    left  = merge_sort(arr[:mid])    # Recursively sort left half
    right = merge_sort(arr[mid:])    # Recursively sort right half

    # -- Conquer (merge the two sorted halves) --------------------------------
    return _merge(left, right)


def _merge(left: list, right: list) -> list:
    """
    Merge two sorted lists into one sorted list.

    Uses two pointers (i into left, j into right) and always appends the
    smaller of the two current elements.  Remaining elements are appended
    via extend once one pointer is exhausted (they are already in order).

    Time Complexity : O(n) where n = len(left) + len(right).
    """
    result, i, j = [], 0, 0

    # Compare the front elements of each half and take the smaller one
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # Append any remaining elements -- exactly one of these will be non-empty
    result.extend(left[i:])
    result.extend(right[j:])
    return result


# ==============================================================================
# SECTION 3 -- Benchmarking Harness
# ==============================================================================

def benchmark(sort_fn, data: list, repetitions: int = 3) -> float:
    """
    Measure the best wall-clock time for ``sort_fn`` on a copy of ``data``.

    We use the *best* of multiple repetitions rather than the mean because
    the minimum filters out noise from OS scheduling, garbage collection,
    and CPU throttling while still reflecting genuine algorithmic cost.
    time.perf_counter() provides the highest available timer resolution.

    Parameters
    ----------
    sort_fn     : callable
        A sorting function with signature f(list) -> list.
    data        : list
        The input data.  A fresh copy is made for each repetition so the
        function always receives an unsorted (or consistently ordered) list.
    repetitions : int
        Number of timing trials.  Default 3 reduces jitter without making
        the total benchmark time excessive.

    Returns
    -------
    float
        Best observed elapsed time in seconds.
    """
    best = float("inf")
    for _ in range(repetitions):
        copy  = data[:]                      # Ensure each trial starts fresh
        start = time.perf_counter()
        sort_fn(copy)
        elapsed = time.perf_counter() - start
        best = min(best, elapsed)            # Keep only the fastest trial
    return best


def _nearly_sorted(n: int, swaps: int = 10) -> list:
    """
    Generate a nearly-sorted list of length n.

    Starts with a fully sorted sequence [0, 1, ..., n-1] and applies
    ``swaps`` random transpositions.  With 10 swaps on a list of 50,000
    elements, the result is 99.98% sorted -- representative of incrementally
    maintained data (e.g., a log file with occasional out-of-order entries).
    """
    arr = list(range(n))
    for _ in range(swaps):
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        arr[i], arr[j] = arr[j], arr[i]    # Apply a single random transposition
    return arr


def run_benchmarks() -> tuple:
    """
    Execute the full benchmark suite and return raw timing data.

    Benchmark Design
    ----------------
    Input sizes       : [500, 1000, 2500, 5000, 10000, 25000, 50000]
    Distributions     : Random, Sorted, Reverse Sorted, Nearly Sorted
    Algorithms        : Heapsort, Quicksort (median-of-three), Merge Sort
    Repetitions/trial : 3  (best time recorded)
    Timer             : time.perf_counter()

    The recursion limit is temporarily raised to n*2 per trial to prevent
    Python stack overflows on large sorted inputs, which push recursive
    Quicksort and Merge Sort to their maximum call depths.

    Returns
    -------
    sizes   : list[int]
        The input sizes tested, in ascending order.
    results : dict[str, dict[str, list[float]]]
        Nested dict: results[distribution][algorithm] = [time_ms, ...]
        One float per size entry, aligned with the returned sizes list.
    """
    sizes = [500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000]

    # Each lambda generates a fresh unsorted list of length n on demand
    distributions = {
        "Random":         lambda n: [random.randint(0, n * 10) for _ in range(n)],
        "Sorted":         lambda n: list(range(n)),
        "Reverse Sorted": lambda n: list(range(n, 0, -1)),
        "Nearly Sorted":  lambda n: _nearly_sorted(n),
    }

    algorithms = {
        "Heapsort":   heapsort,
        "Quicksort":  quicksort,
        "Merge Sort": merge_sort,
    }

    # Pre-allocate the nested result structure
    results = {dist: {alg: [] for alg in algorithms} for dist in distributions}

    for dist_name, gen in distributions.items():
        for n in sizes:
            data = gen(n)       # Generate one dataset per (distribution, size)
            for alg_name, fn in algorithms.items():
                # Temporarily increase recursion depth for deep-recursing sorts
                old_limit = sys.getrecursionlimit()
                sys.setrecursionlimit(max(old_limit, n * 2))

                t = benchmark(fn, data)          # Run timed trial

                sys.setrecursionlimit(old_limit) # Always restore original limit
                results[dist_name][alg_name].append(t * 1_000)  # Convert s -> ms

    return sizes, results


# ==============================================================================
# SECTION 4 -- Script Entry Point (Correctness Verification)
# ==============================================================================

if __name__ == "__main__":
    # -- Correctness check: compare heapsort output with Python's built-in sort
    print("=" * 55)
    print("  Heapsort Correctness Verification")
    print("=" * 55)

    test = [random.randint(0, 100) for _ in range(20)]
    print(f"  Original  : {test}")
    print(f"  Heapsorted: {heapsort(test)}")
    print(f"  Matches sorted(): {heapsort(test) == sorted(test)}")
    print()

    # -- Spot-check edge cases ------------------------------------------------
    print("  Edge cases:")
    print(f"    Empty list  : {heapsort([])}")
    print(f"    Single item : {heapsort([42])}")
    print(f"    All equal   : {heapsort([7, 7, 7, 7])}")
    print(f"    Sorted asc  : {heapsort([1, 2, 3, 4, 5])}")
    print(f"    Sorted desc : {heapsort([5, 4, 3, 2, 1])}")
