"""
interaction_graph.py
--------------------
Bipartite adjacency-list graph for user-product interactions,
with breadth-first search (BFS) for candidate recommendation.

Course : MSCS-532 – Algorithms and Data Structures
Author : Saru Bhandari
"""

from collections import deque


class InteractionGraph:
    """
    Bipartite graph where user nodes and product nodes are connected
    by interaction edges.  Each edge is stored in both directions so
    BFS can start from a user node.

    Time complexity:
        add_interaction        : O(1) amortised
        bfs_recommendations    : O(V + E)
    """

    def __init__(self):
        self.graph = {}   # node -> list of neighbours

    def add_interaction(self, user_id, product_id):
        """Add an undirected edge between user_id and product_id."""
        self.graph.setdefault(user_id, []).append(product_id)
        self.graph.setdefault(product_id, []).append(user_id)

    def get_products(self, user_id):
        """Return the direct neighbours of a user node."""
        return self.graph.get(user_id, [])

    def bfs_recommendations(self, start_user, already_purchased, depth=2):
        """
        Return candidate products reachable within *depth* hops from
        start_user that the user has not already purchased.

        Parameters
        ----------
        start_user       : str   – target user ID
        already_purchased: set   – products to exclude from results
        depth            : int   – maximum hops (default 2)

        Returns
        -------
        set of product IDs
        """
        if start_user not in self.graph:
            return set()

        visited = {start_user}
        queue = deque([(start_user, 0)])
        candidates = set()

        while queue:
            node, d = queue.popleft()
            if d >= depth:
                continue
            for neighbour in self.graph.get(node, []):
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append((neighbour, d + 1))
                    # Only add product nodes (not user nodes) to candidates
                    if not neighbour.startswith("U") and neighbour not in already_purchased:
                        candidates.add(neighbour)

        return candidates


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from user_product_store import UserProductStore

    store = UserProductStore()
    graph = InteractionGraph()

    interactions = [
        ("U1", "P1"), ("U1", "P2"),
        ("U2", "P2"), ("U2", "P3"),
        ("U3", "P3"), ("U3", "P4"),
    ]
    for uid, pid in interactions:
        store.add_interaction(uid, pid)
        graph.add_interaction(uid, pid)

    already_bought = store.get_products("U1")  # {'P1', 'P2'}

    # depth=6 reaches: U1->P2->U2->P3->U3->P4
    candidates = graph.bfs_recommendations("U1", already_bought, depth=6)
    print("Candidates for U1 :", candidates)        # {'P3', 'P4'}

    # Edge case
    print("Unknown user      :", graph.bfs_recommendations("U_UNKNOWN", set()))  # set()
