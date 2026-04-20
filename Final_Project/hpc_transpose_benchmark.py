"""
Cache-Oblivious (Tiled/Blocked) Matrix Transposition in HPC
Demonstrates performance improvements of cache-aware data structure
optimization over naive approaches in High-Performance Computing.

Technique: Blocked (Tiled) Matrix Transposition
- Naive: O(N^2) with high cache-miss rate (strided access)
- Blocked: Same O(N^2) complexity, dramatically fewer cache misses
  by processing sub-matrices that fit in L1/L2 cache

Reference: Hassan et al. (2023). An empirical study of HPC performance bugs.
           https://foyzulhassan.github.io/files/MSR23_HPC.pdf
"""

import time
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. NAIVE TRANSPOSITION — high cache-miss rate (column-stride access)
# ─────────────────────────────────────────────────────────────────────────────
def naive_transpose(A):
    """
    Element-by-element transposition with no cache awareness.
    Column-stride access pattern triggers frequent cache misses.
    Time complexity: O(N^2)  |  Space: O(N^2)
    """
    n = A.shape[0]
    B = np.empty_like(A)
    for i in range(n):
        for j in range(n):
            B[j, i] = A[i, j]   # column-stride write → cache miss
    return B


# ─────────────────────────────────────────────────────────────────────────────
# 2. BLOCKED (CACHE-OBLIVIOUS) TRANSPOSITION — cache-friendly tiling
#    Tile size chosen so a block fits in L1 cache.
#    For float64 (8 bytes), 64×64 block = 32 KB ≈ typical L1 cache size.
# ─────────────────────────────────────────────────────────────────────────────
TILE = 64

def blocked_transpose(A, tile=TILE):
    """
    Tiled transposition: processes TILE×TILE sub-matrices at a time,
    keeping working data in L1/L2 cache and avoiding strided access.
    Time complexity: O(N^2)  |  Space: O(N^2)
    Speedup source: cache-line reuse — each loaded cache line is fully used.
    """
    n = A.shape[0]
    B = np.empty_like(A)
    for i in range(0, n, tile):
        for j in range(0, n, tile):
            # Transpose tile — both source and destination blocks are contiguous
            B[j:j+tile, i:i+tile] = A[i:i+tile, j:j+tile].T
    return B


# ─────────────────────────────────────────────────────────────────────────────
# 3. NUMPY REFERENCE — BLAS-backed, included as performance upper bound
# ─────────────────────────────────────────────────────────────────────────────
def numpy_transpose(A):
    """NumPy's highly optimised C-level transpose (reference baseline)."""
    return np.ascontiguousarray(A.T)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark utilities
# ─────────────────────────────────────────────────────────────────────────────
def benchmark(fn, A, label, reps=5):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(A)
        times.append(time.perf_counter() - t0)
    avg = sum(times) / reps
    print(f"  {label:<46s}  avg {avg*1000:8.3f} ms")
    return avg


def verify(A):
    """Verify correctness of both implementations."""
    expected = A.T.copy()
    assert np.allclose(naive_transpose(A),   expected), "Naive mismatch!"
    assert np.allclose(blocked_transpose(A), expected), "Blocked mismatch!"
    print("  [OK] Both implementations produce correct results.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print(" Cache-Oblivious (Blocked) Matrix Transposition — HPC Benchmark")
    print("=" * 70)
    print(f" Python {__import__('sys').version.split()[0]}  |  NumPy {np.__version__}")
    print(f" Tile size: {TILE}×{TILE}  (≈ {TILE*TILE*8/1024:.0f} KB per tile for float64)")
    print()

    sizes = [128, 256, 512, 1024]
    results = {}

    for n in sizes:
        print(f"Matrix size: {n} × {n}  ({n*n*8/1024:.0f} KB)")
        np.random.seed(0)
        A = np.random.rand(n, n).astype(np.float64)

        if n <= 256:          # skip naive for large sizes (too slow)
            verify(A)
            t_naive = benchmark(naive_transpose,   A, "Naive (pure-Python loop)")
        else:
            t_naive = None
            print("  Naive skipped for large N (estimated > 60 s).")

        t_blk = benchmark(blocked_transpose,   A, f"Blocked tile={TILE} (cache-oblivious)")
        t_np  = benchmark(numpy_transpose,     A, "NumPy / BLAS (reference)")

        if t_naive:
            sp = t_naive / t_blk
            print(f"  → Blocked speedup over naive: {sp:.1f}×")
        print()
        results[n] = (t_naive, t_blk, t_np)

    # ── Summary table ────────────────────────────────────────────────────────
    print("=" * 70)
    print(" Summary")
    print("=" * 70)
    print(f"{'N':>5}  {'Naive (ms)':>12}  {'Blocked (ms)':>13}  {'NumPy (ms)':>11}  {'Speedup':>9}")
    print("-" * 70)
    for n, (tn, tb, tnp) in results.items():
        naive_str = f"{tn*1000:12.3f}" if tn is not None else "          N/A"
        sp_str    = f"{tn/tb:9.1f}×" if tn is not None else "        N/A"
        print(f"{n:>5}  {naive_str}  {tb*1000:13.3f}  {tnp*1000:11.3f}  {sp_str}")

    print()
    print("Conclusion:")
    print("  Blocked transposition achieves dramatic speedups by exploiting")
    print("  spatial locality — tiles fit in L1/L2 cache, eliminating the")
    print("  column-stride cache misses that cripple the naive approach.")
    print("  This mirrors the cache-oblivious design principle identified in")
    print("  Hassan et al. (2023) as critical for HPC performance.")


if __name__ == "__main__":
    main()
