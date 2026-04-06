# Project Deliverable 3 – Optimization, Scaling, and Final Evaluation

**Course:** MSCS-532-B01 – Algorithms and Data Structures (Spring 2026)
**Author:** Saru Bhandari

---

## Files

| File | Description |
|------|-------------|
| `recommendation_system_v2.py` | Phase 3 optimized implementation (all 5 classes + benchmarks) |
| `test_recommendation_system_v2.py` | 35-test pytest suite (unit + integration + stress) |
| `Deliverable3_Phase3.docx` | Full written report (APA format) |

---

## How to Run

**Requirements:** Python 3.8+, no external libraries needed.

```bash
# Run the system demo + benchmarks
python3 recommendation_system_v2.py

# Run all tests (requires pytest)
pytest test_recommendation_system_v2.py -v

# Or without pytest
python3 -m unittest discover  # or run the inline demo in the script
```

---

## Phase 3 Optimizations Summary

| Component | Phase 2 PoC | Phase 3 Optimized |
|-----------|-------------|-------------------|
| Store return type | mutable `set` | `frozenset` (hashable for cache) |
| Graph edge storage | dict of lists, no dedup | dict of lists + `seen_edges` set |
| Graph candidate filter | string prefix heuristic | explicit `_product_nodes` registry |
| Priority queue tie-break | lexicographic (wrong) | insertion-order FIFO (correct) |
| Heap construction | sequential push O(n log n) | `heapify` O(n) |
| Scoring function | hardcoded values | Jaccard similarity + `lru_cache` |

---

## Test Results

35 tests — 35 passed, 0 failed

---

## Key Benchmark Findings

- V2 BFS is 18% faster than PoC at 500-user scale due to edge deduplication
- Full pipeline (BFS + Jaccard scoring + ranking) at 500 users: ~0.51 s for 20 queries
- Jaccard cache hit rate >40% on repeated queries, reducing scoring time ~38%
- Build time is higher in V2 (additional dedup structures) — expected and acceptable trade-off
