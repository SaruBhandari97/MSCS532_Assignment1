"""
test_recommendation_system_v2.py
=================================
Comprehensive test suite for the Phase 3 optimised recommendation system.

Covers:
  - Unit tests for each data structure
  - Integration tests for the full pipeline
  - Edge cases and boundary conditions
  - Stress tests at scale
  - Regression tests verifying Phase 2 behaviour is preserved

Run with:  pytest test_recommendation_system_v2.py -v

Course : MSCS-532-B01 – Algorithms and Data Structures (Spring 2026)
Author : Saru Bhandari
"""

import pytest
import random
import time

from recommendation_system_v2 import (
    UserProductStore,
    InteractionGraph,
    RecommendationQueue,
    JaccardScorer,
    RecommendationPipeline,
    build_dataset,
)


# ══════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════

@pytest.fixture
def small_store():
    """Phase 2 regression dataset: 3 users, 4 products."""
    store = UserProductStore()
    for u, p in [("U1","P1"),("U1","P2"),("U2","P2"),
                 ("U2","P3"),("U3","P3"),("U3","P4")]:
        store.add_interaction(u, p)
    return store


@pytest.fixture
def small_graph(small_store):
    graph = InteractionGraph()
    for u, p in [("U1","P1"),("U1","P2"),("U2","P2"),
                 ("U2","P3"),("U3","P3"),("U3","P4")]:
        graph.add_interaction(u, p)
    return graph


@pytest.fixture
def pipeline(small_store, small_graph):
    return RecommendationPipeline(small_store, small_graph)


# ══════════════════════════════════════════════════════════════════════
# 1.  UserProductStore unit tests
# ══════════════════════════════════════════════════════════════════════

class TestUserProductStore:

    def test_add_and_retrieve_forward(self, small_store):
        """Forward lookup: user → products."""
        assert small_store.get_products("U1") == frozenset({"P1", "P2"})

    def test_add_and_retrieve_reverse(self, small_store):
        """Reverse lookup: product → users."""
        assert small_store.get_users("P2") == frozenset({"U1", "U2"})

    def test_duplicate_interaction_ignored(self, small_store):
        """Adding the same edge twice must not inflate the set."""
        before = len(small_store.get_products("U1"))
        small_store.add_interaction("U1", "P1")   # duplicate
        assert len(small_store.get_products("U1")) == before

    def test_interaction_count_no_duplicates(self):
        """interaction_count reflects unique edges only."""
        store = UserProductStore()
        store.add_interaction("U1", "P1")
        store.add_interaction("U1", "P1")   # duplicate
        store.add_interaction("U1", "P2")
        assert store.interaction_count == 2

    def test_unknown_user_returns_empty(self, small_store):
        """Querying unknown user must return empty frozenset, not raise."""
        assert small_store.get_products("U_GHOST") == frozenset()

    def test_unknown_product_returns_empty(self, small_store):
        assert small_store.get_users("P_GHOST") == frozenset()

    def test_return_type_is_frozenset(self, small_store):
        """get_products must return frozenset (hashable for lru_cache)."""
        result = small_store.get_products("U1")
        assert isinstance(result, frozenset)

    def test_counters(self, small_store):
        assert small_store.user_count == 3
        assert small_store.product_count == 4


# ══════════════════════════════════════════════════════════════════════
# 2.  InteractionGraph unit tests
# ══════════════════════════════════════════════════════════════════════

class TestInteractionGraph:

    def test_bfs_excludes_already_purchased(self, small_graph, small_store):
        """Candidates must not include products U1 already bought."""
        already = small_store.get_products("U1")
        candidates = small_graph.bfs_recommendations("U1", already, depth=4)
        assert "P1" not in candidates
        assert "P2" not in candidates

    def test_bfs_finds_correct_candidates(self, small_graph, small_store):
        """BFS at depth 4 must surface P3 and P4 for U1."""
        already = small_store.get_products("U1")
        candidates = small_graph.bfs_recommendations("U1", already, depth=4)
        assert {"P3", "P4"}.issubset(candidates)

    def test_bfs_unknown_user_returns_empty(self, small_graph):
        candidates = small_graph.bfs_recommendations("U_GHOST", frozenset(), depth=4)
        assert candidates == set()

    def test_edge_deduplication(self):
        """Duplicate add_interaction calls must not create duplicate neighbours."""
        graph = InteractionGraph()
        graph.add_interaction("U1", "P1")
        graph.add_interaction("U1", "P1")   # duplicate
        assert graph.edge_count == 1
        # Adjacency list should contain P1 exactly once for U1
        assert graph._adj["U1"].count("P1") == 1

    def test_candidates_are_product_nodes_only(self, small_graph, small_store):
        """BFS must not include user IDs in the candidate set."""
        already = small_store.get_products("U1")
        candidates = small_graph.bfs_recommendations("U1", already, depth=4)
        for c in candidates:
            assert c.startswith("P"), f"User node {c!r} leaked into candidates"

    def test_node_and_edge_counts(self, small_graph):
        # 3 users + 4 products = 7 nodes; 6 unique interactions
        assert small_graph.node_count == 7
        assert small_graph.edge_count == 6


# ══════════════════════════════════════════════════════════════════════
# 3.  RecommendationQueue unit tests
# ══════════════════════════════════════════════════════════════════════

class TestRecommendationQueue:

    def test_pop_returns_highest_score(self):
        rq = RecommendationQueue()
        rq.push(0.85, "P3")
        rq.push(0.62, "P4")
        rq.push(0.91, "P5")
        pid, score = rq.pop_top()
        assert pid == "P5"
        assert abs(score - 0.91) < 1e-9

    def test_get_top_n_correct_order(self):
        rq = RecommendationQueue()
        rq.push(0.85, "P3")
        rq.push(0.62, "P4")
        rq.push(0.91, "P5")
        top = rq.get_top_n(3)
        scores = [s for _, s in top]
        assert scores == sorted(scores, reverse=True)

    def test_get_top_n_non_destructive(self):
        """Queue size must be unchanged after get_top_n."""
        rq = RecommendationQueue()
        for i, pid in enumerate(["P1","P2","P3"]):
            rq.push(float(i), pid)
        before = len(rq)
        rq.get_top_n(2)
        assert len(rq) == before

    def test_bulk_load_matches_sequential_push(self):
        """bulk_load and sequential push must produce identical top-n."""
        items = [(0.9,"PA"),(0.7,"PB"),(0.5,"PC"),(0.3,"PD")]

        rq1 = RecommendationQueue()
        for score, pid in items:
            rq1.push(score, pid)

        rq2 = RecommendationQueue()
        rq2.bulk_load(items)

        assert rq1.get_top_n(4) == rq2.get_top_n(4)

    def test_tie_breaking_by_insertion_order(self):
        """Tied scores must be broken by insertion order (FIFO)."""
        rq = RecommendationQueue()
        rq.push(0.9, "P_first")
        rq.push(0.9, "P_second")
        pid, _ = rq.pop_top()
        assert pid == "P_first"

    def test_pop_empty_returns_none(self):
        rq = RecommendationQueue()
        assert rq.pop_top() == (None, None)

    def test_peek_top_non_destructive(self):
        rq = RecommendationQueue()
        rq.push(0.8, "P1")
        rq.peek_top()
        assert len(rq) == 1


# ══════════════════════════════════════════════════════════════════════
# 4.  JaccardScorer unit tests
# ══════════════════════════════════════════════════════════════════════

class TestJaccardScorer:

    def test_score_range(self, small_store):
        scorer = JaccardScorer(small_store)
        score = scorer.score("U1", "P3")
        assert 0.0 <= score <= 1.0

    def test_high_similarity_score(self):
        """When target and buyer share most products, Jaccard approaches 1."""
        store = UserProductStore()
        # U1 and U2 share P1, P2 – U2 also has P3
        store.add_interaction("U1", "P1")
        store.add_interaction("U1", "P2")
        store.add_interaction("U2", "P1")
        store.add_interaction("U2", "P2")
        store.add_interaction("U2", "P3")
        scorer = JaccardScorer(store)
        # Jaccard({P1,P2}, {P1,P2,P3}) = 2/3 ≈ 0.667
        score = scorer.score("U1", "P3")
        assert abs(score - 2/3) < 1e-9

    def test_disjoint_sets_score_zero(self):
        store = UserProductStore()
        store.add_interaction("U1", "P1")
        store.add_interaction("U2", "P99")
        scorer = JaccardScorer(store)
        score = scorer.score("U1", "P99")
        assert abs(score - 0.0) < 1e-9

    def test_cache_hits(self, small_store):
        """Repeated calls must generate cache hits."""
        scorer = JaccardScorer(small_store)
        scorer.score("U1", "P3")
        scorer.score("U1", "P3")   # should hit cache
        info = scorer.cache_info()
        assert info.hits >= 1

    def test_unknown_product_returns_zero(self, small_store):
        scorer = JaccardScorer(small_store)
        assert scorer.score("U1", "P_GHOST") == 0.0


# ══════════════════════════════════════════════════════════════════════
# 5.  RecommendationPipeline integration tests
# ══════════════════════════════════════════════════════════════════════

class TestRecommendationPipeline:

    def test_recommendations_exclude_purchased(self, pipeline, small_store):
        recs = pipeline.recommend("U1", top_n=5)
        purchased = small_store.get_products("U1")
        for pid, _ in recs:
            assert pid not in purchased

    def test_recommendations_sorted_descending(self, pipeline):
        recs = pipeline.recommend("U1", top_n=5)
        scores = [s for _, s in recs]
        assert scores == sorted(scores, reverse=True)

    def test_unknown_user_returns_empty_list(self, pipeline):
        assert pipeline.recommend("U_GHOST") == []

    def test_top_n_limit_respected(self, pipeline):
        recs = pipeline.recommend("U1", top_n=2)
        assert len(recs) <= 2

    def test_all_recommended_products_are_products(self, pipeline):
        """No user IDs should appear in recommendations."""
        recs = pipeline.recommend("U1", top_n=5)
        for pid, _ in recs:
            assert pid.startswith("P")


# ══════════════════════════════════════════════════════════════════════
# 6.  Stress tests
# ══════════════════════════════════════════════════════════════════════

class TestStress:

    @pytest.mark.parametrize("n_users,n_products", [
        (500,  1000),
        (1000, 2000),
    ])
    def test_build_and_recommend_at_scale(self, n_users, n_products):
        """Pipeline must complete within 5 seconds for medium-scale datasets."""
        store, graph = build_dataset(n_users, n_products, interactions_per_user=15)
        pipeline = RecommendationPipeline(store, graph)

        t0 = time.perf_counter()
        # Sample 20 users for recommendations
        for u in range(min(20, n_users)):
            pipeline.recommend(f"U{u}", top_n=5)
        elapsed = time.perf_counter() - t0

        assert elapsed < 5.0, f"Pipeline too slow at scale: {elapsed:.2f}s"

    def test_store_handles_large_interaction_volume(self):
        """Store must not corrupt data under 50k interactions."""
        store = UserProductStore()
        rng = random.Random(0)
        n = 50_000
        for _ in range(n):
            u = f"U{rng.randint(0, 999)}"
            p = f"P{rng.randint(0, 1999)}"
            store.add_interaction(u, p)
        # Every stored interaction should be retrievable in reverse
        for uid in store.all_users():
            for pid in store.get_products(uid):
                assert uid in store.get_users(pid)

    def test_priority_queue_bulk_load_large(self):
        """bulk_load must handle 10,000 items without error."""
        rq = RecommendationQueue()
        items = [(random.random(), f"P{i}") for i in range(10_000)]
        rq.bulk_load(items)
        assert len(rq) == 10_000
        # Top score must be the maximum from items
        _, top_score = rq.peek_top()
        assert abs(top_score - max(s for s, _ in items)) < 1e-9
