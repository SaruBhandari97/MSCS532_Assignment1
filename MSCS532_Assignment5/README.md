# Assignment 5 – Quicksort: Implementation, Analysis, and Randomization

**Course:** MSCS-532-B01 – Algorithms and Data Structures (Spring 2026)
**Author:** Saru Bhandari

---

## Files

| File | Description |
|------|-------------|
| `quicksort.py` | Deterministic and randomized Quicksort implementations + benchmarks |
| `Assignment5_Quicksort.docx` | Full written report (APA format) |

---

## How to Run

**Requirements:** Python 3.8+, no external libraries needed.

```bash
python3 quicksort.py
```

This will:
1. Run a correctness check on a small sample array
2. Print a benchmark table comparing both algorithms across three input distributions (random, sorted, reverse-sorted) and three sizes (500, 1,000, 2,000)

---

## Summary of Findings

- **Deterministic Quicksort** (last-element pivot) degrades to O(n²) on sorted and reverse-sorted inputs. At n = 2,000 on sorted data it was ~88× slower than on random data.
- **Randomized Quicksort** (random pivot) maintains O(n log n) expected performance on all input types. On sorted input at n = 2,000 it ran in 0.0024 s vs. 0.1466 s for the deterministic version.
- On random input both versions perform similarly, with deterministic slightly faster due to no random-number overhead.

---

## Complexity Reference

| Case | Deterministic | Randomized |
|------|--------------|------------|
| Best | O(n log n) | O(n log n) |
| Average | O(n log n) | O(n log n) expected |
| Worst | O(n²) | O(n²) with negligible probability |
| Space | O(log n) avg | O(log n) expected |
