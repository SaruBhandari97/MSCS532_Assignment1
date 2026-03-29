"""
recommendation_queue.py
-----------------------
Max-heap priority queue for ranking product recommendations.

Python's heapq is a min-heap, so scores are stored as negatives to
simulate max-heap behaviour.  Each heap entry is:
    (-score, product_id)

Time complexity:
    push             : O(log n)
    pop_top          : O(log n)
    get_top_n        : O(n log n)

Course : MSCS-532 – Algorithms and Data Structures
Author : Saru Bhandari
"""

import heapq


class RecommendationQueue:
    """Max-heap priority queue for ranked product recommendations."""

    def __init__(self):
        self.heap = []

    def push(self, score, product_id):
        """Add product_id with its relevance score."""
        heapq.heappush(self.heap, (-score, product_id))

    def pop_top(self):
        """Remove and return the (product_id, score) with the highest score."""
        if self.heap:
            neg_score, product_id = heapq.heappop(self.heap)
            return product_id, -neg_score
        return None, None

    def get_top_n(self, n):
        """
        Return the top n (product_id, score) pairs without modifying the queue.
        """
        popped, results = [], []
        for _ in range(min(n, len(self.heap))):
            item = heapq.heappop(self.heap)
            results.append((item[1], -item[0]))
            popped.append(item)
        for item in popped:
            heapq.heappush(self.heap, item)
        return results


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rq = RecommendationQueue()

    rq.push(0.85, "P3")
    rq.push(0.62, "P4")
    rq.push(0.91, "P5")

    print("Top 3 (non-destructive) :", rq.get_top_n(3))
    # [('P5', 0.91), ('P3', 0.85), ('P4', 0.62)]

    product, score = rq.pop_top()
    print("Popped                  :", product, score)   # P5  0.91
    print("Next top after pop      :", rq.get_top_n(1))  # [('P3', 0.85)]

    # Empty queue
    empty = RecommendationQueue()
    print("Pop from empty          :", empty.pop_top())  # (None, None)
