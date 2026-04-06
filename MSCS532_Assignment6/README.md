# Assignment 6: Medians and Order Statistics & Elementary Data Structures

**Course:** MSCS-532-B01 – Algorithms and Data Structures  
**Author:** Saru Bhandari  
**Reference:** Cormen, Leiserson, Rivest & Stein — *Introduction to Algorithms*, 4th Edition (ISBN: 9780262046305)

---

## Repository Structure

```
├── part1_selection_algorithms.py   # Median of Medians + Randomized Quickselect
├── part2_data_structures.py        # Dynamic Array, Matrix, Stack, Queue, Linked List, Rooted Tree
└── README.md                       # This file
```

---

## How to Run

**Requirements:** Python 3.8 or later. No external libraries needed.

```bash
# Part 1 — runs correctness checks then prints a benchmark table
python part1_selection_algorithms.py

# Part 2 — runs a live demonstration of all six data structures
python part2_data_structures.py
```

---

## Part 1: Selection Algorithms

### What is implemented

| Algorithm | Worst-case | Expected | CLRS Reference |
|---|---|---|---|
| Median of Medians (deterministic) | O(n) | O(n) | §9.3 |
| Randomized Quickselect | O(n²) | O(n) | §9.2 |

### Key design decisions

- **Three-way partition** in Median of Medians correctly handles arrays with duplicate elements — a standard two-way Lomuto partition would misplace equal elements.
- **Non-destructive** — both public functions copy the input so the caller's array is never modified.
- **Correctness verifier** built in — on startup, both algorithms are checked against Python's `sorted()` across five test cases including duplicates, single elements, and large random arrays.

### Public API

```python
from part1_selection_algorithms import deterministic_select, randomized_select

arr = [3, 1, 4, 1, 5, 9, 2, 6]

# k is 1-indexed: k=1 is minimum, k=len(arr) is maximum
print(deterministic_select(arr, 3))   # → 2  (3rd smallest)
print(randomized_select(arr, 3))      # → 2
```

### Empirical results summary

Both algorithms scale linearly. Randomized Quickselect is consistently **4–7× faster** due to lower constant factors. Neither algorithm degrades on sorted or reverse-sorted inputs.

| Distribution | n | Deterministic (ms) | Randomized (ms) |
|---|---|---|---|
| Random | 10,000 | ~13 | ~3 |
| Random | 100,000 | ~138 | ~24 |
| Sorted | 100,000 | ~96 | ~20 |
| Reverse-sorted | 100,000 | ~141 | ~28 |

---

## Part 2: Elementary Data Structures

### What is implemented

| Class | Description | CLRS Reference |
|---|---|---|
| `DynamicArray` | Resizable array, amortized O(1) append via doubling | §17.4 |
| `Matrix` | 2-D array in row-major flat storage, O(1) access | §10.1 |
| `Stack` | Array-backed LIFO, all ops O(1) amortized | §10.1 |
| `Queue` | Circular ring buffer FIFO, O(1) amortized enqueue/dequeue | §10.1 |
| `SinglyLinkedList` | Sentinel-head list with tail pointer, O(1) push/append | §10.2 |
| `RootedTree` | Left-child / right-sibling general tree | §10.4 |

### Quick usage examples

```python
from part2_data_structures import Stack, Queue, SinglyLinkedList, RootedTree

# Stack
s = Stack()
s.push(10); s.push(20)
print(s.pop())        # → 20

# Queue
q = Queue()
q.enqueue("a"); q.enqueue("b")
print(q.dequeue())    # → "a"

# Singly Linked List
ll = SinglyLinkedList()
ll.append(1); ll.append(2); ll.push_front(0)
print(ll.traverse())  # → [0, 1, 2]

# Rooted Tree
t = RootedTree("root")
child = t.add_child(t.root, "child1")
t.add_child(child, "grandchild")
print(t.preorder())   # → ['root', 'child1', 'grandchild']
print(t.height())     # → 2
```

### Complexity summary

| Structure | Key operations | Time |
|---|---|---|
| Dynamic Array | access, append | O(1) / O(1) amortized |
| Dynamic Array | insert, delete | O(n) |
| Matrix | access, set | O(1) |
| Stack | push, pop, peek | O(1) amortized |
| Queue | enqueue, dequeue | O(1) amortized |
| Linked List | push_front, append | O(1) |
| Linked List | search, delete | O(n) |
| Rooted Tree | preorder, postorder | O(n) |
| Rooted Tree | height, depth | O(n) / O(h) |

---

## Summary of Findings

### Part 1
- Both algorithms run in **linear time** — confirmed empirically across all input distributions.
- **Randomized Quickselect** is the practical default: simpler code, smaller constant, and O(n) expected time on any input.
- **Median of Medians** is preferable when worst-case guarantees are mandatory (adversarial inputs, real-time systems).

### Part 2
- **Array-backed** stacks and queues outperform linked-list-backed equivalents in cache-sensitive workloads due to memory locality.
- The **circular ring buffer** queue eliminates the O(n) front-deletion penalty of naive array queues.
- The **left-child/right-sibling** tree representation handles arbitrary branching factors in O(n) space with no wasted pointer slots.
