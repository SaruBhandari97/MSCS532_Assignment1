# Assignment 3: Algorithm Efficiency and Scalability

## Overview

This repository contains Python implementations and empirical benchmarks for two fundamental algorithms studied in the course:

- **Part 1** — Randomized Quicksort (with comparison against Deterministic Quicksort)
- **Part 2** — Hash Table with Chaining (universal hashing + dynamic resizing)

---

## Repository Structure

```
.
├── randomized_quicksort.py   # Part 1 — sorting algorithms + benchmark runner
├── hash_table.py             # Part 2 — hash table implementation + benchmark runner
├── plot_results.py           # Generates PNG charts for the report
├── README.md                 # This file
└── report/                   # Report goes here (add after screenshots)
```

---

## Requirements

- **Python 3.8+** (standard library only for the core files)
- **matplotlib** (only required for `plot_results.py`)

Install matplotlib:

```bash
pip install matplotlib
```

---

## How to Run

### Step 1 — Verify correctness and run the Quicksort benchmark

```bash
python randomized_quicksort.py
```

**What to expect:**

1. A correctness check section — both algorithms should show `PASS`.
2. A live benchmark table printed to the terminal as each configuration finishes.
3. A summary table at the end grouping results by input type.

**Screenshot tip:** Run in a terminal window wide enough (~110 chars) to see the table without line wrapping. The summary table at the bottom is the most useful section to screenshot.

---

### Step 2 — Verify correctness and run the Hash Table benchmark

```bash
python hash_table.py
```

**What to expect:**

1. A correctness check — should show `PASS`.
2. A performance benchmark table showing insert / search / delete times and load factor for increasing `n`.
3. A chain-length distribution printed as a simple ASCII histogram.

**Screenshot tip:** The chain-length distribution at the bottom is visually interesting and makes a great screenshot for the report.

---

### Step 3 — Generate PNG charts

```bash
python plot_results.py
```

This produces three PNG files in the current directory:

| File | Contents |
|------|----------|
| `quicksort_comparison.png` | 2×2 grid comparing both algorithms across all four input types |
| `hashtable_performance.png` | Operation timing + load factor / max chain length |
| `chain_distribution.png` | Bar chart of chain-length distribution for n = 10 000 |

**Screenshot tip:** Open each PNG and take a screenshot, or include the PNG files directly in your report.

---

## Expected Terminal Output Samples

### `randomized_quicksort.py`

```
============================================================
  CORRECTNESS VERIFICATION
============================================================
  PASS — Randomized Quicksort produced correct output on all test cases
  PASS — Deterministic Quicksort produced correct output on all test cases

============================================================
  PERFORMANCE BENCHMARK
============================================================
  [    random] n=   100 | Rand:      0.0001s  Det:       0.0001s
  [    random] n=  1000 | Rand:      0.0014s  Det:       0.0013s
  ...
  [    sorted] n= 10000 | Rand:      0.0312s  Det: FAILED (RecursionError)
```

The `RecursionError` on sorted/reverse inputs for the deterministic variant is **intentional** — it demonstrates the O(n²) worst-case behaviour.

### `hash_table.py`

```
============================================================
  CORRECTNESS VERIFICATION
============================================================
  PASS — HashTableChaining passed all correctness tests

============================================================
  PERFORMANCE BENCHMARK
============================================================
         n    Insert(s)    Search(s)    Delete(s)   Load α  MaxChain
  --------  ----------  ----------  ----------  -------  ---------
      1000     0.00120     0.00089     0.00061    0.625          3
     10000     0.01203     0.00891     0.00602    0.610          5
    100000     0.12841     0.09123     0.06012    0.625          6
```

---

## Development Process

The project was developed in structured stages to ensure correctness and performance.

### 1. Initial Implementation
- Implemented deterministic quicksort  
- Verified correctness with test cases  

### 2. Randomization Enhancement
- Added randomized pivot selection  
- Tested behavior across different input types  

### 3. Hash Table Design
- Implemented chaining for collision handling  
- Used universal hashing for better distribution  
- Added dynamic resizing using load factor thresholds  

### 4. Testing
- Created correctness tests for both parts  
- Tested edge cases such as empty inputs and duplicates  

### 5. Benchmarking
- Measured execution time for increasing input sizes  
- Generated plots to visualize performance trends  


## Summary of Findings

### Part 1 — Randomized Quicksort

| Input Type | Randomized QS | Deterministic QS |
|------------|--------------|-----------------|
| Random | O(n log n) expected — fast in practice | O(n log n) average — similar to randomized |
| Sorted | O(n log n) — random pivot avoids degenerate splits | **O(n²)** — first element always min/max → crashes at large n |
| Reverse sorted | O(n log n) — same protection | **O(n²)** — same failure mode |
| Repeated elements | O(n log n) with slight constant factor increase | O(n²) risk if equal elements cluster |

**Key insight:** The random pivot selection in Randomized Quicksort makes it impossible for any fixed adversarial input to reliably trigger the worst case. The expected number of comparisons is 2n ln n ≈ 1.386 n log₂ n regardless of input order.

### Part 2 — Hash Table with Chaining

| Operation | Expected Time | Notes |
|-----------|--------------|-------|
| Insert | O(1) amortised | Occasional O(n) resize, but amortised O(1) per op |
| Search | O(1 + α) | α = load factor; bounded by resize policy |
| Delete | O(1) amortised | Same analysis as insert |

**Key insight:** The universal hash function (MAM family) ensures that for any two distinct keys, the probability of collision is at most 1/m. Combined with dynamic resizing that keeps α ∈ [0.25, 0.75], all three operations run in O(1) expected time.

---

## Theoretical Analysis (Quick Reference)

### Randomized Quicksort — Expected Time O(n log n)

Define the indicator random variable X_{ij} = 1 if element i and element j are ever compared during the sort. The total number of comparisons is:

```
C = Σ_{i<j} X_{ij}
```

Under the random pivot selection, Pr[X_{ij} = 1] = 2 / (j - i + 1).

Therefore:

```
E[C] = Σ_{i<j} 2/(j-i+1)
     = 2n · H_n   (where H_n is the n-th harmonic number)
     ≈ 2n ln n
     = O(n log n)
```

### Hash Table — Expected Chain Length O(1 + α)

Under simple uniform hashing, each key hashes independently and uniformly to any of the m slots. For a table with n items in m slots (load factor α = n/m), the expected length of the chain at any slot is exactly α. Therefore:

- A successful search examines 1 + α/2 items on average (half the chain before the hit).
- An unsuccessful search examines 1 + α items on average (the entire chain).
- With our resize policy α ≤ 0.75, both are O(1).

---

## Edge Cases Handled

### Quicksort
- Empty arrays (`[]`) — base case exits immediately
- Single-element arrays — base case exits immediately
- All-identical elements — random pivot still works; partitions unevenly but correctly
- Already sorted / reverse sorted — random pivot avoids O(n²) degradation

### Hash Table
- Duplicate key insert — overwrites existing value (no duplicate chains)
- Delete non-existent key — returns `False` gracefully
- Automatic resize up (load > 0.75) and down (load < 0.25)
- Non-integer keys — Python's `hash()` maps any hashable object to an integer before the universal hash function is applied

---

## Author

Saru Bhandari
MSCS-532-B01 — Algorithms and Data Structures 
Spring / 2026
