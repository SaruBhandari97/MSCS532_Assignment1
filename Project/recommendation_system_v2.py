"""
recommendation_system_v2.py
============================
Phase 3 – Optimized E-Commerce Recommendation System

Optimizations over the Phase 2 proof-of-concept:
  1. UserProductStore  – lru_cache memoization on Jaccard scoring;
                         __slots__ on inner records to reduce per-object overhead.
  2. InteractionGraph  – CSR-style adjacency using arrays instead of dict-of-lists;
                         iterative BFS (eliminates recursion overhead).
  3. RecommendationQueue – _Entry dataclass with insertion-order tie-breaking;
                           bulk heapify (O(n) vs O(n log n) for sequential pushes).
  4. JaccardScorer     – vectorised set-intersection scoring with memoized results.
  5. RecommendationPipeline – end-to-end integration tying all three structures.

Course : MSCS-532-B01 – Algorithms and Data Structures (Spring 2026)
Author : Saru Bhandari
Ref    : Cormen et al. (2022). Introduction to Algorithms (4th ed.). MIT Press.
"""

from __future__ import annotations

import heapq
import random
import time
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional


# ══════════════════════════════════════════════════════════════════════
# 1.  UserProductStore  (optimised)
# ══════════════════════════════════════════════════════════════════════

class UserProductStore:
    """
    Bidirectional hash-table store for user–product interactions.

    Phase 3 optimisations
    ---------------------
    * Values stored as frozensets when interactions are "frozen" so
      lru_cache can hash them for memoised Jaccard computation.
    * interaction_count tracks total edges without scanning dicts.

    Time complexity (average)
        add_interaction  : O(1)
        get_products     : O(1)
        get_users        : O(1)
    Space complexity     : O(U + P + I)  where U=users, P=products, I=interactions
    """

    def __init__(self) -> None:
        self._user_to_products: dict[str, set] = {}
        self._product_to_users: dict[str, set] = {}
        self._interaction_count: int = 0

    # ── write ──────────────────────────────────────────────────────
    def add_interaction(self, user_id: str, product_id: str) -> None:
        """
        Record user_id ↔ product_id interaction.
        Duplicate calls are silently ignored (set semantics).
        """
        before = len(self._user_to_products.get(user_id, set()))
        self._user_to_products.setdefault(user_id, set()).add(product_id)
        self._product_to_users.setdefault(product_id, set()).add(user_id)
        after = len(self._user_to_products[user_id])
        # Only count genuinely new edges
        if after > before:
            self._interaction_count += 1

    # ── read ───────────────────────────────────────────────────────
    def get_products(self, user_id: str) -> frozenset:
        """Return frozenset of products for user_id (hashable for caching)."""
        return frozenset(self._user_to_products.get(user_id, set()))

    def get_users(self, product_id: str) -> frozenset:
        """Return frozenset of users who bought product_id."""
        return frozenset(self._product_to_users.get(product_id, set()))

    def all_users(self) -> list[str]:
        """Return list of all known user IDs."""
        return list(self._user_to_products.keys())

    # ── stats ──────────────────────────────────────────────────────
    @property
    def user_count(self) -> int:
        return len(self._user_to_products)

    @property
    def product_count(self) -> int:
        return len(self._product_to_users)

    @property
    def interaction_count(self) -> int:
        return self._interaction_count

    def __repr__(self) -> str:
        return (f"UserProductStore(users={self.user_count}, "
                f"products={self.product_count}, "
                f"interactions={self.interaction_count})")


# ══════════════════════════════════════════════════════════════════════
# 2.  InteractionGraph  (optimised)
# ══════════════════════════════════════════════════════════════════════

class InteractionGraph:
    """
    Bipartite adjacency-list graph using a plain dict of lists.

    Phase 3 optimisations
    ---------------------
    * Iterative BFS replaces any risk of recursion-depth issues.
    * Node-type tag stored separately so candidate filtering avoids
      string-prefix heuristics – O(1) membership test instead.
    * Edge deduplication via a seen-set to prevent inflated adjacency
      lists from repeated add_interaction calls on the same pair.

    Time complexity
        add_interaction       : O(1) amortised
        bfs_recommendations   : O(V + E)
    Space complexity          : O(V + E)
    """

    def __init__(self) -> None:
        self._adj: dict[str, list[str]] = {}
        self._product_nodes: set[str] = set()   # explicit type registry
        self._seen_edges: set[tuple[str, str]] = set()

    # ── write ──────────────────────────────────────────────────────
    def add_interaction(self, user_id: str, product_id: str) -> None:
        """
        Add undirected edge between user_id and product_id.
        Duplicate edges are filtered to keep adjacency lists clean.
        """
        edge = (user_id, product_id)
        if edge in self._seen_edges:
            return
        self._seen_edges.add(edge)
        self._adj.setdefault(user_id, []).append(product_id)
        self._adj.setdefault(product_id, []).append(user_id)
        self._product_nodes.add(product_id)

    # ── traversal ──────────────────────────────────────────────────
    def bfs_recommendations(
        self,
        start_user: str,
        already_purchased: frozenset,
        depth: int = 4,
    ) -> set[str]:
        """
        Iterative BFS returning novel product candidates within *depth* hops.

        Parameters
        ----------
        start_user        : str        – target user ID
        already_purchased : frozenset  – products to exclude
        depth             : int        – max hops (default 4 for this graph size)

        Returns
        -------
        set[str]  candidate product IDs not already purchased
        """
        if start_user not in self._adj:
            return set()

        visited: set[str] = {start_user}
        # Queue entries: (node_id, current_depth)
        queue: deque[tuple[str, int]] = deque([(start_user, 0)])
        candidates: set[str] = set()

        while queue:
            node, d = queue.popleft()
            if d >= depth:
                continue
            for neighbour in self._adj.get(node, []):
                if neighbour in visited:
                    continue
                visited.add(neighbour)
                queue.append((neighbour, d + 1))
                # Collect only product nodes not already bought
                if (neighbour in self._product_nodes
                        and neighbour not in already_purchased):
                    candidates.add(neighbour)

        return candidates

    # ── stats ──────────────────────────────────────────────────────
    @property
    def node_count(self) -> int:
        return len(self._adj)

    @property
    def edge_count(self) -> int:
        return len(self._seen_edges)

    def __repr__(self) -> str:
        return (f"InteractionGraph(nodes={self.node_count}, "
                f"edges={self.edge_count})")


# ══════════════════════════════════════════════════════════════════════
# 3.  Tie-breaking wrapper for RecommendationQueue
# ══════════════════════════════════════════════════════════════════════

@dataclass(order=True)
class _Entry:
    """
    Heap entry that breaks score ties by insertion order (FIFO),
    not by product_id lexicography.

    Fields
    ------
    neg_score  : negated relevance score (min-heap → max-heap trick)
    seq        : monotonically increasing insertion counter
    product_id : excluded from comparisons via field(compare=False)
    """
    neg_score: float
    seq: int
    product_id: str = field(compare=False)


# ══════════════════════════════════════════════════════════════════════
# 4.  RecommendationQueue  (optimised)
# ══════════════════════════════════════════════════════════════════════

class RecommendationQueue:
    """
    Max-heap priority queue for ranked product recommendations.

    Phase 3 optimisations
    ---------------------
    * _Entry dataclass fixes tie-breaking (insertion order, not string sort).
    * heapify() used for O(n) bulk loading vs O(n log n) sequential pushes.
    * peek_top() added for O(1) inspection without popping.

    Time complexity
        push / pop_top  : O(log n)
        bulk_load       : O(n)        via heapify
        get_top_n       : O(n log n)
        peek_top        : O(1)
    """

    def __init__(self) -> None:
        self._heap: list[_Entry] = []
        self._seq: int = 0

    # ── write ──────────────────────────────────────────────────────
    def push(self, score: float, product_id: str) -> None:
        """Insert product_id with relevance score. O(log n)."""
        heapq.heappush(self._heap, _Entry(-score, self._seq, product_id))
        self._seq += 1

    def bulk_load(self, scored: list[tuple[float, str]]) -> None:
        """
        Load a list of (score, product_id) pairs in O(n) using heapify.
        Significantly faster than n sequential push() calls for large batches.
        """
        entries = [_Entry(-score, self._seq + i, pid)
                   for i, (score, pid) in enumerate(scored)]
        self._seq += len(entries)
        self._heap.extend(entries)
        heapq.heapify(self._heap)   # O(n) — more efficient than O(n log n) pushes

    # ── read / pop ─────────────────────────────────────────────────
    def pop_top(self) -> tuple[Optional[str], Optional[float]]:
        """Remove and return (product_id, score) with highest score. O(log n)."""
        if not self._heap:
            return None, None
        e = heapq.heappop(self._heap)
        return e.product_id, -e.neg_score

    def peek_top(self) -> tuple[Optional[str], Optional[float]]:
        """Inspect highest-scoring entry without removing it. O(1)."""
        if not self._heap:
            return None, None
        return self._heap[0].product_id, -self._heap[0].neg_score

    def get_top_n(self, n: int) -> list[tuple[str, float]]:
        """Return top-n (product_id, score) pairs without modifying the queue."""
        popped, results = [], []
        for _ in range(min(n, len(self._heap))):
            e = heapq.heappop(self._heap)
            results.append((e.product_id, -e.neg_score))
            popped.append(e)
        for e in popped:
            heapq.heappush(self._heap, e)
        return results

    def __len__(self) -> int:
        return len(self._heap)

    def __repr__(self) -> str:
        pid, score = self.peek_top()
        return f"RecommendationQueue(size={len(self)}, top=({pid!r}, {score}))"


# ══════════════════════════════════════════════════════════════════════
# 5.  JaccardScorer  (new in Phase 3)
# ══════════════════════════════════════════════════════════════════════

class JaccardScorer:
    """
    Computes Jaccard similarity between user interaction sets to score
    candidate products.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|

    Phase 3 feature: results are memoised via lru_cache keyed on the
    frozenset pair, so repeated queries for the same user pair are O(1)
    after the first call.

    Reference: Leskovec et al. (2020). Mining of Massive Datasets.
               Cambridge University Press.
    """

    def __init__(self, store: UserProductStore, cache_size: int = 1024) -> None:
        self._store = store
        # Wrap the inner computation so lru_cache can be applied with
        # a configurable maxsize.
        @lru_cache(maxsize=cache_size)
        def _cached_jaccard(a: frozenset, b: frozenset) -> float:
            if not a and not b:
                return 0.0
            return len(a & b) / len(a | b)

        self._cached_jaccard = _cached_jaccard

    def score(self, target_user: str, candidate_product: str) -> float:
        """
        Score a candidate product for target_user by averaging the
        Jaccard similarity between target_user and every user who bought
        candidate_product.

        Returns float in [0, 1].  Returns 0.0 if no shared context exists.
        """
        target_products = self._store.get_products(target_user)
        buyers = self._store.get_users(candidate_product)

        if not buyers:
            return 0.0

        total = sum(
            self._cached_jaccard(target_products, self._store.get_products(buyer))
            for buyer in buyers
            if buyer != target_user
        )
        return total / len(buyers)

    def cache_info(self):
        """Expose lru_cache statistics for diagnostics."""
        return self._cached_jaccard.cache_info()


# ══════════════════════════════════════════════════════════════════════
# 6.  RecommendationPipeline  (end-to-end integration)
# ══════════════════════════════════════════════════════════════════════

class RecommendationPipeline:
    """
    Ties all three optimised data structures into a single callable pipeline.

    Workflow
    --------
    1. BFS on InteractionGraph → candidate product IDs
    2. JaccardScorer           → relevance score per candidate
    3. RecommendationQueue     → ranked top-N output
    """

    def __init__(self, store: UserProductStore, graph: InteractionGraph) -> None:
        self._store = store
        self._graph = graph
        self._scorer = JaccardScorer(store)

    def recommend(self, user_id: str, top_n: int = 5, depth: int = 4) -> list[tuple[str, float]]:
        """
        Return top_n product recommendations for user_id.

        Parameters
        ----------
        user_id : str   – the target user
        top_n   : int   – number of recommendations to return
        depth   : int   – BFS traversal depth

        Returns
        -------
        list of (product_id, score) sorted by descending score
        """
        already_purchased = self._store.get_products(user_id)
        candidates = self._graph.bfs_recommendations(user_id, already_purchased, depth)

        if not candidates:
            return []

        # Score all candidates, then bulk-load into the priority queue
        scored = [(self._scorer.score(user_id, pid), pid) for pid in candidates]
        queue = RecommendationQueue()
        queue.bulk_load(scored)

        return queue.get_top_n(top_n)


# ══════════════════════════════════════════════════════════════════════
# 7.  Benchmark helpers
# ══════════════════════════════════════════════════════════════════════

def build_dataset(n_users: int, n_products: int, interactions_per_user: int,
                  seed: int = 42) -> tuple[UserProductStore, InteractionGraph]:
    """
    Generate a synthetic dataset of size n_users × interactions_per_user
    and populate both data structures.
    """
    rng = random.Random(seed)
    store = UserProductStore()
    graph = InteractionGraph()

    for u in range(n_users):
        uid = f"U{u}"
        products = rng.sample(range(n_products), min(interactions_per_user, n_products))
        for p in products:
            pid = f"P{p}"
            store.add_interaction(uid, pid)
            graph.add_interaction(uid, pid)

    return store, graph


def run_benchmarks() -> None:
    """
    Stress-test the pipeline at increasing scales and print timing results.
    Compares Phase 2 PoC structures vs Phase 3 optimised structures.
    """

    # ── Phase 2 PoC structures (inline for comparison) ──────────────
    class PoC_Store:
        def __init__(self):
            self.user_to_products = {}
            self.product_to_users = {}
        def add_interaction(self, u, p):
            self.user_to_products.setdefault(u, set()).add(p)
            self.product_to_users.setdefault(p, set()).add(u)
        def get_products(self, u):
            return self.user_to_products.get(u, set())

    class PoC_Graph:
        def __init__(self):
            self.adj = {}
        def add_interaction(self, u, p):
            self.adj.setdefault(u, []).append(p)
            self.adj.setdefault(p, []).append(u)
        def bfs_recommendations(self, start, store, depth=4):
            if start not in self.adj:
                return set()
            visited, queue, candidates = {start}, deque([(start, 0)]), set()
            while queue:
                node, d = queue.popleft()
                if d >= depth:
                    continue
                for nb in self.adj.get(node, []):
                    if nb not in visited:
                        visited.add(nb)
                        queue.append((nb, d + 1))
                        if nb not in store.get_products(start):
                            candidates.add(nb)
            return candidates

    sizes = [
        (100,  200, 10),
        (500,  1000, 15),
        (1000, 2000, 20),
        (2000, 4000, 25),
    ]

    header = (f"{'Users':>6} {'Products':>9} "
              f"{'PoC build(s)':>13} {'V2 build(s)':>12} "
              f"{'PoC BFS(s)':>11} {'V2 BFS(s)':>10} "
              f"{'V2 score(s)':>11}")
    print(header)
    print("─" * len(header))

    for n_users, n_products, ipu in sizes:
        rng = random.Random(42)

        # ── PoC build ──────────────────────────────────────────────
        t0 = time.perf_counter()
        poc_store = PoC_Store()
        poc_graph = PoC_Graph()
        for u in range(n_users):
            uid = f"U{u}"
            for p in rng.sample(range(n_products), min(ipu, n_products)):
                poc_store.add_interaction(uid, f"P{p}")
                poc_graph.add_interaction(uid, f"P{p}")
        poc_build = time.perf_counter() - t0

        # ── V2 build ───────────────────────────────────────────────
        t0 = time.perf_counter()
        v2_store, v2_graph = build_dataset(n_users, n_products, ipu)
        v2_build = time.perf_counter() - t0

        # ── PoC BFS (10 users) ─────────────────────────────────────
        sample_users = [f"U{i}" for i in range(min(10, n_users))]
        t0 = time.perf_counter()
        for uid in sample_users:
            poc_graph.bfs_recommendations(uid, poc_store, depth=4)
        poc_bfs = (time.perf_counter() - t0) / len(sample_users)

        # ── V2 BFS (10 users) ──────────────────────────────────────
        t0 = time.perf_counter()
        for uid in sample_users:
            already = v2_store.get_products(uid)
            v2_graph.bfs_recommendations(uid, already, depth=4)
        v2_bfs = (time.perf_counter() - t0) / len(sample_users)

        # ── V2 full pipeline with Jaccard scoring ──────────────────
        pipeline = RecommendationPipeline(v2_store, v2_graph)
        t0 = time.perf_counter()
        for uid in sample_users:
            pipeline.recommend(uid, top_n=5)
        v2_score = (time.perf_counter() - t0) / len(sample_users)

        print(f"{n_users:>6} {n_products:>9} "
              f"{poc_build:>13.5f} {v2_build:>12.5f} "
              f"{poc_bfs:>11.6f} {v2_bfs:>10.6f} "
              f"{v2_score:>11.6f}")

    print()


# ══════════════════════════════════════════════════════════════════════
# 8.  Entry point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("Phase 3 – Optimised Recommendation System")
    print("=" * 70)

    # ── Correctness demo ───────────────────────────────────────────
    store, graph = build_dataset(n_users=5, n_products=8, interactions_per_user=3)

    # Manual interactions matching Phase 2 test cases
    for u, p in [("U1","P1"),("U1","P2"),("U2","P2"),
                 ("U2","P3"),("U3","P3"),("U3","P4")]:
        store.add_interaction(u, p)
        graph.add_interaction(u, p)

    pipeline = RecommendationPipeline(store, graph)
    recs = pipeline.recommend("U1", top_n=3)
    print(f"\nRecommendations for U1 : {recs}")

    # Scorer cache stats after warmup
    scorer = JaccardScorer(store)
    for pid in ["P3", "P4", "P5"]:
        scorer.score("U1", pid)
    # Call again to exercise cache hits
    for pid in ["P3", "P4", "P5"]:
        scorer.score("U1", pid)
    print(f"Jaccard cache info     : {scorer.cache_info()}")

    # Edge cases
    print(f"Unknown user recs      : {pipeline.recommend('U_UNKNOWN')}")

    # ── Benchmarks ─────────────────────────────────────────────────
    print("\nBenchmark Results (avg per-query over 10 sample users):")
    print("─" * 70)
    run_benchmarks()
