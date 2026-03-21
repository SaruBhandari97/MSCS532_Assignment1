"""
hash_table.py
=============
Hash table using chaining for collision handling,
with dynamic resizing and simple performance benchmarking.

Assignment 3 - Part 2
"""

import random
import time


# ---------------------------------------------------------------------------
# Hash Table Implementation
# ---------------------------------------------------------------------------

class HashTableChaining:
    """
    Hash table using chaining (each slot stores a list of key-value pairs).

    Uses a universal hash function to reduce collisions and
    resizes automatically to maintain performance.
    """

    # Load factor thresholds
    LOAD_HIGH = 0.75   # grow table
    LOAD_LOW  = 0.25   # shrink table
    MIN_SLOTS = 8

    # Large prime for hashing
    _PRIME = (1 << 31) - 1

    def __init__(self, initial_slots=8):
        """Initialize table with empty chains."""
        self._m = initial_slots
        self._n = 0
        self._buckets = [[] for _ in range(self._m)]
        self._pick_hash_params()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_hash_params(self):
        """Pick random parameters for the hash function."""
        p = self._PRIME
        self._a = random.randint(1, p - 1)
        self._b = random.randint(0, p - 1)

    def _hash(self, key):
        """Compute slot index for a given key."""
        k = hash(key) & 0x7FFFFFFF
        return ((self._a * k + self._b) % self._PRIME) % self._m

    def _resize(self, new_m):
        """Rebuild table with a new number of slots."""
        old_buckets = self._buckets

        self._m = new_m
        self._buckets = [[] for _ in range(self._m)]
        self._n = 0
        self._pick_hash_params()

        # Reinsert existing elements
        for chain in old_buckets:
            for key, value in chain:
                slot = self._hash(key)
                self._buckets[slot].append((key, value))
                self._n += 1

    def _maybe_resize(self):
        """Check load factor and resize if needed."""
        load = self._n / self._m

        if load > self.LOAD_HIGH:
            self._resize(self._m * 2)
        elif load < self.LOAD_LOW and self._m > self.MIN_SLOTS:
            self._resize(max(self._m // 2, self.MIN_SLOTS))

    # ------------------------------------------------------------------
    # Public operations
    # ------------------------------------------------------------------

    def insert(self, key, value):
        """Insert or update a key-value pair."""
        slot = self._hash(key)
        chain = self._buckets[slot]

        # Update if key exists
        for i, (k, v) in enumerate(chain):
            if k == key:
                chain[i] = (key, value)
                return

        # Otherwise insert new entry
        chain.append((key, value))
        self._n += 1
        self._maybe_resize()

    def search(self, key):
        """Return value for key, or None if not found."""
        slot = self._hash(key)
        chain = self._buckets[slot]

        for k, v in chain:
            if k == key:
                return v
        return None

    def delete(self, key):
        """Remove key if present. Returns True/False."""
        slot = self._hash(key)
        chain = self._buckets[slot]

        for i, (k, v) in enumerate(chain):
            if k == key:
                del chain[i]
                self._n -= 1
                self._maybe_resize()
                return True
        return False

    # ------------------------------------------------------------------
    # Utility properties
    # ------------------------------------------------------------------

    @property
    def load_factor(self):
        return self._n / self._m

    @property
    def size(self):
        return self._n

    @property
    def num_slots(self):
        return self._m

    def chain_lengths(self):
        """Return list of chain sizes."""
        return [len(b) for b in self._buckets]

    def max_chain_length(self):
        """Return longest chain length."""
        return max(self.chain_lengths())

    def __repr__(self):
        return f"HashTableChaining(n={self._n}, m={self._m}, load={self.load_factor:.3f})"


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def benchmark_operations(n_items):
    """
    Benchmark insert, search, and delete operations.
    """
    ht = HashTableChaining()

    keys = list(range(n_items))
    values = [f"val_{k}" for k in keys]
    random.shuffle(keys)

    # Insert
    t0 = time.perf_counter()
    for k, v in zip(keys, values):
        ht.insert(k, v)
    insert_time = time.perf_counter() - t0

    load_after_insert = ht.load_factor
    max_chain = ht.max_chain_length()

    # Search
    t0 = time.perf_counter()
    hits = sum(1 for k in keys if ht.search(k) is not None)
    search_time = time.perf_counter() - t0

    # Delete half
    delete_keys = keys[: n_items // 2]
    t0 = time.perf_counter()
    for k in delete_keys:
        ht.delete(k)
    delete_time = time.perf_counter() - t0

    load_after_delete = ht.load_factor

    return {
        "n": n_items,
        "insert_time": insert_time,
        "search_time": search_time,
        "delete_time": delete_time,
        "search_hits": hits,
        "load_after_insert": load_after_insert,
        "load_after_delete": load_after_delete,
        "max_chain_length": max_chain,
        "slots_after_insert": ht.num_slots,
    }


# ---------------------------------------------------------------------------
# Correctness testing
# ---------------------------------------------------------------------------

def verify_hash_table():
    """Basic correctness tests."""
    ht = HashTableChaining()

    ht.insert("apple", 1)
    ht.insert("banana", 2)
    ht.insert("cherry", 3)

    assert ht.search("apple") == 1
    assert ht.search("banana") == 2
    assert ht.search("cherry") == 3
    assert ht.search("missing") is None

    # Update
    ht.insert("apple", 99)
    assert ht.search("apple") == 99
    assert ht.size == 3

    # Delete
    assert ht.delete("banana") is True
    assert ht.search("banana") is None
    assert ht.size == 2

    assert ht.delete("unknown") is False

    # Stress test
    ht2 = HashTableChaining()
    for i in range(1000):
        ht2.insert(i, i * 2)
    for i in range(1000):
        assert ht2.search(i) == i * 2

    print("  PASS — Hash table works correctly")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  CORRECTNESS VERIFICATION")
    print("=" * 60)
    verify_hash_table()

    print()
    print("=" * 60)
    print("  PERFORMANCE BENCHMARK")
    print("=" * 60)

    print(f"\n  {'n':>8}  {'Insert(s)':>10}  {'Search(s)':>10}  "
          f"{'Delete(s)':>10}  {'Load α':>7}  {'MaxChain':>9}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*7}  {'-'*9}")

    for n in [1_000, 5_000, 10_000, 50_000, 100_000]:
        r = benchmark_operations(n)
        print(f"  {r['n']:>8}  "
              f"{r['insert_time']:>10.5f}  "
              f"{r['search_time']:>10.5f}  "
              f"{r['delete_time']:>10.5f}  "
              f"{r['load_after_insert']:>7.3f}  "
              f"{r['max_chain_length']:>9}")

    print("\n  Note: resizing keeps load factor under control.")

    # Quick distribution check
    print("\n" + "=" * 60)
    print("  CHAIN LENGTH DISTRIBUTION (n = 10,000)")
    print("=" * 60)

    ht = HashTableChaining()
    for i in range(10_000):
        ht.insert(i, i)

    from collections import Counter
    dist = Counter(ht.chain_lengths())

    print(f"\n  Slots: {ht.num_slots}, Items: {ht.size}, Load: {ht.load_factor:.3f}")
    print(f"  Max chain length: {ht.max_chain_length()}")

    for length in sorted(dist):
        bar = "#" * min(dist[length], 60)
        print(f"    length {length:>2}: {dist[length]:>5}  {bar}")