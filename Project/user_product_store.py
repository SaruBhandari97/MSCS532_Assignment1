"""
user_product_store.py
---------------------
Hash-table-based store for user-product interactions.

Course : MSCS-532 – Algorithms and Data Structures
Author : Saru Bhandari
"""


class UserProductStore:
    """
    Stores user-product interactions using two bidirectional hash maps.

    Time complexity (average case):
        add_interaction : O(1)
        get_products    : O(1)
        get_users       : O(1)
    """

    def __init__(self):
        self.user_to_products = {}   # user_id  -> set of product_ids
        self.product_to_users = {}   # product_id -> set of user_ids

    def add_interaction(self, user_id, product_id):
        """Record that user_id interacted with product_id."""
        self.user_to_products.setdefault(user_id, set()).add(product_id)
        self.product_to_users.setdefault(product_id, set()).add(user_id)

    def get_products(self, user_id):
        """Return all products a user has interacted with."""
        return self.user_to_products.get(user_id, set())

    def get_users(self, product_id):
        """Return all users who interacted with a product."""
        return self.product_to_users.get(product_id, set())


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    store = UserProductStore()

    store.add_interaction("U1", "P1")
    store.add_interaction("U1", "P2")
    store.add_interaction("U2", "P2")
    store.add_interaction("U2", "P3")
    store.add_interaction("U3", "P3")
    store.add_interaction("U3", "P4")

    # Duplicate — should have no effect
    store.add_interaction("U1", "P1")

    print("U1's products :", store.get_products("U1"))       # {'P1', 'P2'}
    print("P2's users    :", store.get_users("P2"))           # {'U1', 'U2'}
    print("Unknown user  :", store.get_products("U_UNKNOWN")) # set()
