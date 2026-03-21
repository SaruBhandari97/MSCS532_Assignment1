# Assignment 4 – Heap Data Structures: Implementation, Analysis, and Applications

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Process Narrative](#process-narrative)
4. [How to Run the Code](#how-to-run-the-code)
5. [Implementation Details](#implementation-details)
6. [Complexity Analysis Summary](#complexity-analysis-summary)
7. [Benchmark Results and Key Findings](#benchmark-results-and-key-findings)
8. [Design Decisions and Tradeoffs](#design-decisions-and-tradeoffs)
9. [References](#references)

---

## Project Overview

This assignment explores heap data structures from three angles: building one correctly from scratch, analyzing why it behaves the way it does, and applying it to a real scheduling problem. The two core deliverables are a **Heapsort implementation** and a **max-heap priority queue**, each accompanied by empirical benchmarks and a formal APA-style report.

---

## Repository Structure

```
heap-assignment/
├── heapsort.py           # Heapsort + Quicksort + Merge Sort + benchmarking harness
├── priority_queue.py     # Task class + MaxHeapPriorityQueue + scheduler simulation
├── generate_charts.py    # Runs benchmarks and produces the four chart PNGs
├── charts/
│   ├── chart1_distribution_comparison.png
│   ├── chart2_loglog_random.png
│   ├── chart3_pathological.png
│   └── chart4_pq_operations.png
├── report_APA7.docx      # Full written report in APA 7 format
└── README.md             # This file
```

---

## Process Narrative

### Where I Started: Understanding the Heap Invariant

Before writing any code I spent time making sure I understood the heap property rigorously — not just that a parent is "bigger than its children," but *why* the array representation works. The key insight is that the parent–child relationship is encoded arithmetically: for a node at index `i`, the left child is at `2i + 1` and the right child at `2i + 2`. This means the entire tree structure lives implicitly in a flat Python list — no pointers, no Node objects, no extra memory.

Once I understood that, the `heapify` function became intuitive: it is simply "look at me and my two children, put the biggest one on top, and if I moved anything, recurse into the subtree I just changed."

### Implementing Heapsort

I implemented Heapsort in three layers, each building on the previous:

1. **`heapify(arr, n, i)`** — the single fundamental operation. It assumes both subtrees of node `i` are already valid heaps and fixes only the root. I initially made the mistake of calling it with `n = len(arr)` during the extraction phase, which caused already-sorted elements at the tail to be reconsidered. The fix was to pass `i` (the shrinking heap boundary) rather than the full array length.

2. **`build_max_heap(arr)`** — calls `heapify` on every non-leaf node from the bottom up. The counterintuitive result is that this is O(n), not O(n log n), because most calls happen near the leaves where subtrees are tiny. I verified this by reading the geometric series proof in Cormen et al. (2022) and confirming the experimental timing scaled linearly.

3. **`heapsort(arr)`** — assembles the two phases. One important detail: I made the function work on a copy of the input (`arr[:]`) so callers are never surprised by mutation.

### Adding Comparison Algorithms

The assignment asked for an empirical comparison, so I implemented **Quicksort** and **Merge Sort** as baselines. I used a **median-of-three pivot** for Quicksort specifically to make the comparison fair — naive Quicksort degrades to O(n²) on sorted inputs, which would have made the results misleading on those distributions. I wanted to compare the *best reasonable* version of each algorithm, not a strawman.

I also had to temporarily raise Python's recursion limit for large sorted inputs, since recursive Quicksort and Merge Sort can reach depths of up to n on a pre-sorted array. I made sure the limit was restored after each trial so other parts of the program were unaffected.

### Designing the Priority Queue

The biggest design decision for the priority queue was whether to include an **auxiliary position map** (`_pos` dict). The standard textbook heap does not include one, which means `increase_key` and `decrease_key` require a linear scan to find the target task — O(n) rather than O(log n).

I decided to include it because it makes the data structure genuinely useful. Without it, a priority queue cannot efficiently support re-prioritization, which is exactly what operating system schedulers and graph algorithms like Dijkstra's do constantly. The cost is a small amount of extra memory (one dict entry per task) and the discipline to update `_pos` on every swap. I solved the latter by centralizing all swaps in a single `_swap()` method that always updates both the heap array and the position map atomically — no other method ever directly swaps elements.

### Benchmarking Methodology

I wrote a `benchmark()` function that takes the **best of three repetitions** rather than the mean. The reasoning is that the *minimum* time is closest to the true algorithmic cost — it filters out noise from OS scheduling, garbage collection pauses, and CPU frequency scaling. Using the mean would penalize algorithms unfairly when a single unlucky trial happened to collide with a GC pause.

Benchmark results were saved to `benchmark_results.json` and then loaded by `generate_charts.py` to produce the figures. This separation means the charts can be regenerated without re-running the (slow) benchmarks.

### Generating the Charts

Each of the four charts was designed to answer a specific question:

| Chart | Question it answers |
|---|---|
| Chart 1 (4-panel) | Does the performance pattern differ across input distributions? |
| Chart 2 (log-log) | Do all algorithms actually follow O(n log n) scaling? |
| Chart 3 (bar) | How do the algorithms behave on adversarial (sorted/reverse) inputs? |
| Chart 4 (line) | Do priority queue operations scale as O(log n) in practice? |

The log-log chart (Chart 2) was particularly informative — on log-log axes, O(n log n) growth appears nearly linear (since log(n log n) ≈ log n for large n), and all three empirical curves run parallel to the reference line, confirming the theoretical prediction.

### Writing the Report

The report was structured in APA 7 format with a title page, abstract, two main sections (Heapsort and Priority Queue), a Discussion, Conclusion, and References. Figures and tables were embedded directly into the Word document. All timing data in the report came from actual benchmark runs, not approximations.

---

## How to Run the Code

### Requirements

```
Python 3.8+
matplotlib   (pip install matplotlib)
```

### 1 — Verify Heapsort correctness

```bash
python heapsort.py
```

Expected output: the test array printed in sorted order, edge case results, and a confirmation that the output matches Python's built-in `sorted()`.

### 2 — Run the priority queue scheduler demo

```bash
python priority_queue.py
```

Expected output: step-by-step log showing tasks being inserted, reprioritized, and extracted in priority order.

### 3 — Regenerate benchmark data and charts

```bash
python generate_charts.py
```

This will:
- Run all sorting benchmarks (500–50,000 elements, 4 distributions, 3 algorithms)
- Time priority queue operations across heap sizes 500–10,000
- Save four PNG charts to the `charts/` folder

> **Note:** The benchmark run takes approximately 2–5 minutes depending on your machine. Larger sizes (25,000 and 50,000) are the most time-consuming.

---

## Implementation Details

### heapsort.py

| Function | Role | Complexity |
|---|---|---|
| `heapify(arr, n, i)` | Sift element at `i` downward to restore heap property | O(log n) |
| `build_max_heap(arr)` | Convert array to max-heap bottom-up | O(n) |
| `heapsort(arr)` | Full sort: build heap then extract n times | O(n log n) |
| `quicksort(arr)` | Median-of-three Quicksort (benchmark baseline) | O(n log n) avg |
| `merge_sort(arr)` | Top-down recursive Merge Sort (benchmark baseline) | O(n log n) |
| `benchmark(fn, data)` | Time a sort function, best of 3 reps | — |
| `run_benchmarks()` | Full suite across all sizes and distributions | — |

### priority_queue.py

| Component | Role |
|---|---|
| `Task` dataclass | Represents a schedulable unit of work; compares by priority |
| `MaxHeapPriorityQueue._heap` | Backing list for the binary max-heap |
| `MaxHeapPriorityQueue._pos` | Dict mapping task_id → heap index for O(1) lookup |
| `_swap(i, j)` | Atomic swap that updates both `_heap` and `_pos` |
| `_sift_up(i)` | Bubble element upward after insert / increase_key |
| `_sift_down(i)` | Sink element downward after extract / decrease_key |
| `insert(task)` | Add task, sift up — O(log n) |
| `extract_max()` | Remove and return highest priority task — O(log n) |
| `increase_key(id, p)` | Raise priority, sift up — O(log n) |
| `decrease_key(id, p)` | Lower priority, sift down — O(log n) |
| `peek()` | Read max without removal — O(1) |
| `is_empty()` | Check if queue has no tasks — O(1) |

---

## Complexity Analysis Summary

### Heapsort

The O(n log n) guarantee comes from two phases:

- **build_max_heap**: O(n). Although heapify is called n/2 times, nodes near the leaves have tiny subtrees. Summing over all levels with a geometric series argument collapses the total to linear.
- **Extraction loop**: O(n log n). Each of the n−1 extractions calls heapify at the root, which traverses at most ⌊log n⌋ levels.

Total: O(n) + O(n log n) = **O(n log n), unconditionally**. Unlike Quicksort, there is no worst-case divergence — the heap structure enforces this bound regardless of input order.

### Priority Queue Operations

All key operations are O(log n) because sift-up and sift-down traverse a path from a node to the root or a leaf, and the height of a complete binary tree with n nodes is exactly ⌊log₂ n⌋. The position map makes `increase_key` and `decrease_key` O(log n) rather than O(n) by removing the linear search step.

---

## Benchmark Results and Key Findings

| Finding | Explanation |
|---|---|
| Quicksort fastest on random data | Superior cache behavior and low constant factor, despite same O(n log n) bound |
| Heapsort most consistent across all distributions | No dependency on input ordering; heap height determined by size alone |
| All three algorithms follow O(n log n) scaling | Confirmed by log-log chart; all curves parallel to the reference line |
| Merge Sort uses most memory | Only algorithm requiring O(n) auxiliary space; visible in runtime overhead for large n |
| Priority queue operations scale sub-linearly | Per-operation time grows slowly with heap size, consistent with O(log n) |

---

## Design Decisions and Tradeoffs

**Array vs. linked heap:** Arrays were chosen because they eliminate pointer overhead, enable cache-friendly sequential access, and encode tree structure for free through index arithmetic.

**Max-heap vs. min-heap:** A max-heap was chosen to match the "highest priority first" semantics of preemptive schedulers. A min-heap would be appropriate for shortest-job-first or earliest-deadline-first scheduling policies.

**Position map inclusion:** The `_pos` dict adds O(n) extra memory but reduces `increase_key` and `decrease_key` from O(n) to O(log n), making the structure suitable for use in graph algorithms like Dijkstra's shortest path.

**Recursive heapify:** The recursive implementation is cleaner and easier to reason about, but uses O(log n) call-stack space. An iterative version achieves O(1) auxiliary space at the cost of slightly more code.

---

## References

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to algorithms* (4th ed.). MIT Press.
- Sedgewick, R., & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley Professional.
- Williams, J. W. J. (1964). Algorithm 232: Heapsort. *Communications of the ACM, 7*(6), 347–348.
- Knuth, D. E. (1997). *The art of computer programming, Vol. 3: Sorting and searching* (2nd ed.). Addison-Wesley.
