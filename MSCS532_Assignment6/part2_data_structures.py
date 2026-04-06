"""
Assignment 6 - Part 2: Elementary Data Structures
Course: Introduction to Algorithms (CLRS 4th Edition)

Implements from scratch (no standard-library collections used):
  - DynamicArray     – resizable array with amortised O(1) append
  - Matrix           – 2-D array with O(1) access, row/col operations
  - Stack            – LIFO structure backed by DynamicArray
  - Queue            – FIFO structure using a circular array (ring buffer)
  - SinglyLinkedList – pointer-based list with O(1) push/pop-front
  - RootedTree       – general tree using left-child / right-sibling representation

References:
  - CLRS §10.1  (Stacks and queues)
  - CLRS §10.2  (Linked lists)
  - CLRS §10.4  (Rooted trees)
"""

from __future__ import annotations
from typing import Any, Optional, Iterator


# ════════════════════════════════════════════════════════
#  1.  DYNAMIC ARRAY
# ════════════════════════════════════════════════════════

class DynamicArray:
    """
    A resizable array that doubles capacity when full and halves it
    when fewer than 25 % of slots are used (prevents thrashing).

    Complexities
    ------------
    Access   : O(1)
    Append   : O(1) amortised  (potential-function argument, CLRS §17.4)
    Insert   : O(n)  worst-case (elements must shift right)
    Delete   : O(n)  worst-case (elements must shift left)
    """

    _INITIAL_CAPACITY = 4

    def __init__(self) -> None:
        self._capacity: int = self._INITIAL_CAPACITY
        self._size: int = 0
        self._data: list = [None] * self._capacity

    # ── internal helpers ──────────────────────────────────

    def _resize(self, new_cap: int) -> None:
        """Copy live elements into a fresh backing store of size new_cap."""
        new_data = [None] * new_cap
        for i in range(self._size):
            new_data[i] = self._data[i]
        self._data = new_data
        self._capacity = new_cap

    def _check_index(self, idx: int) -> int:
        """Normalise negative indices and raise on out-of-range."""
        if idx < 0:
            idx += self._size
        if not (0 <= idx < self._size):
            raise IndexError(f"Index {idx} out of range [0, {self._size - 1}].")
        return idx

    # ── public interface ──────────────────────────────────

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> Any:
        return self._data[self._check_index(idx)]

    def __setitem__(self, idx: int, value: Any) -> None:
        self._data[self._check_index(idx)] = value

    def __repr__(self) -> str:
        items = [str(self._data[i]) for i in range(self._size)]
        return "DynamicArray([" + ", ".join(items) + "])"

    def append(self, value: Any) -> None:
        """O(1) amortised – double capacity when full."""
        if self._size == self._capacity:
            self._resize(2 * self._capacity)
        self._data[self._size] = value
        self._size += 1

    def insert(self, idx: int, value: Any) -> None:
        """O(n) – shift elements right to make room."""
        if idx < 0:
            idx += self._size
        idx = max(0, min(idx, self._size))  # clamp to valid insertion point
        if self._size == self._capacity:
            self._resize(2 * self._capacity)
        for i in range(self._size, idx, -1):
            self._data[i] = self._data[i - 1]
        self._data[idx] = value
        self._size += 1

    def delete(self, idx: int) -> Any:
        """O(n) – shift elements left and shrink if < 25 % utilisation."""
        idx = self._check_index(idx)
        removed = self._data[idx]
        for i in range(idx, self._size - 1):
            self._data[i] = self._data[i + 1]
        self._data[self._size - 1] = None   # let GC collect the object
        self._size -= 1
        if self._size > 0 and self._size <= self._capacity // 4:
            self._resize(self._capacity // 2)
        return removed

    def __iter__(self) -> Iterator:
        for i in range(self._size):
            yield self._data[i]


# ════════════════════════════════════════════════════════
#  2.  MATRIX
# ════════════════════════════════════════════════════════

class Matrix:
    """
    A 2-D array stored in row-major order using a flat DynamicArray.

    Access pattern: element at (r, c) → flat index r * cols + c.

    Complexities
    ------------
    Access/Set : O(1)
    Add row    : O(cols)
    Add column : O(rows × cols)  – must shift every row
    Delete row : O(rows × cols)
    """

    def __init__(self, rows: int, cols: int, default: Any = 0) -> None:
        if rows < 0 or cols < 0:
            raise ValueError("Dimensions must be non-negative.")
        self._rows = rows
        self._cols = cols
        self._data = DynamicArray()
        for _ in range(rows * cols):
            self._data.append(default)

    # ── accessors ─────────────────────────────────────────

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    def _flat(self, r: int, c: int) -> int:
        if not (0 <= r < self._rows and 0 <= c < self._cols):
            raise IndexError(f"({r}, {c}) out of bounds ({self._rows}×{self._cols}).")
        return r * self._cols + c

    def get(self, r: int, c: int) -> Any:
        return self._data[self._flat(r, c)]

    def set(self, r: int, c: int, value: Any) -> None:
        self._data[self._flat(r, c)] = value

    # ── structural operations ─────────────────────────────

    def add_row(self, values: Optional[list] = None, default: Any = 0) -> None:
        """Append a new row.  O(cols)."""
        if values is None:
            values = [default] * self._cols
        if len(values) != self._cols:
            raise ValueError("Row length must match number of columns.")
        for v in values:
            self._data.append(v)
        self._rows += 1

    def delete_row(self, r: int) -> None:
        """Remove row r.  O(rows × cols) due to shifts in the flat array."""
        if not (0 <= r < self._rows):
            raise IndexError(f"Row {r} out of range.")
        start = r * self._cols
        for _ in range(self._cols):
            self._data.delete(start)
        self._rows -= 1

    def __repr__(self) -> str:
        lines = []
        for r in range(self._rows):
            row = [str(self._data[r * self._cols + c]) for c in range(self._cols)]
            lines.append("  [" + ", ".join(row) + "]")
        return "Matrix([\n" + "\n".join(lines) + "\n])"


# ════════════════════════════════════════════════════════
#  3.  STACK  (array-backed LIFO)
# ════════════════════════════════════════════════════════

class Stack:
    """
    LIFO stack backed by a DynamicArray (CLRS §10.1).

    All operations are O(1) amortised.

    Attribute   | Complexity
    ------------|-------------
    push        | O(1) amortised
    pop         | O(1) amortised
    peek        | O(1)
    is_empty    | O(1)
    """

    def __init__(self) -> None:
        self._data = DynamicArray()

    def push(self, value: Any) -> None:
        """Push value onto the top of the stack."""
        self._data.append(value)

    def pop(self) -> Any:
        """Remove and return the top element.  Raises IndexError if empty."""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._data.delete(len(self._data) - 1)

    def peek(self) -> Any:
        """Return the top element without removing it."""
        if self.is_empty():
            raise IndexError("peek at empty stack")
        return self._data[len(self._data) - 1]

    def is_empty(self) -> bool:
        return len(self._data) == 0

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"Stack(top → {list(self._data)[::-1]})"


# ════════════════════════════════════════════════════════
#  4.  QUEUE  (circular-array FIFO)
# ════════════════════════════════════════════════════════

class Queue:
    """
    FIFO queue using a circular (ring) buffer (CLRS §10.1).

    A fixed-capacity internal array is doubled when full, giving
    amortised O(1) enqueue and O(1) dequeue.

    Attribute   | Complexity
    ------------|-------------
    enqueue     | O(1) amortised
    dequeue     | O(1) amortised
    peek_front  | O(1)
    is_empty    | O(1)
    """

    _INITIAL_CAPACITY = 8

    def __init__(self) -> None:
        self._capacity: int = self._INITIAL_CAPACITY
        self._buf: list = [None] * self._capacity
        self._head: int = 0        # index of the front element
        self._size: int = 0

    def _resize(self, new_cap: int) -> None:
        """Copy elements in logical order into a larger buffer."""
        new_buf = [None] * new_cap
        for i in range(self._size):
            new_buf[i] = self._buf[(self._head + i) % self._capacity]
        self._buf = new_buf
        self._head = 0
        self._capacity = new_cap

    def enqueue(self, value: Any) -> None:
        """Add value to the rear of the queue.  O(1) amortised."""
        if self._size == self._capacity:
            self._resize(2 * self._capacity)
        tail = (self._head + self._size) % self._capacity
        self._buf[tail] = value
        self._size += 1

    def dequeue(self) -> Any:
        """Remove and return the front element.  Raises IndexError if empty."""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        value = self._buf[self._head]
        self._buf[self._head] = None
        self._head = (self._head + 1) % self._capacity
        self._size -= 1
        if self._size > 0 and self._size <= self._capacity // 4:
            self._resize(max(self._INITIAL_CAPACITY, self._capacity // 2))
        return value

    def peek_front(self) -> Any:
        if self.is_empty():
            raise IndexError("peek at empty queue")
        return self._buf[self._head]

    def is_empty(self) -> bool:
        return self._size == 0

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        items = [self._buf[(self._head + i) % self._capacity]
                 for i in range(self._size)]
        return f"Queue(front → {items})"


# ════════════════════════════════════════════════════════
#  5.  SINGLY LINKED LIST
# ════════════════════════════════════════════════════════

class _SLLNode:
    """Internal node for SinglyLinkedList."""
    __slots__ = ("data", "next")

    def __init__(self, data: Any) -> None:
        self.data: Any = data
        self.next: Optional[_SLLNode] = None


class SinglyLinkedList:
    """
    Singly linked list with a sentinel head and a tail pointer for
    O(1) append (CLRS §10.2).

    Attribute         | Complexity
    ------------------|----------------------------
    push_front        | O(1)
    pop_front         | O(1)
    append            | O(1)   (tail pointer)
    search            | O(n)
    delete (by value) | O(n)
    traverse          | O(n)
    """

    def __init__(self) -> None:
        # Sentinel node: its .next points to the first real node (or None).
        self._sentinel = _SLLNode(None)
        self._tail: Optional[_SLLNode] = None
        self._size: int = 0

    # ── O(1) operations ───────────────────────────────────

    def push_front(self, value: Any) -> None:
        """Insert at the front.  O(1)."""
        node = _SLLNode(value)
        node.next = self._sentinel.next
        self._sentinel.next = node
        if self._tail is None:
            self._tail = node
        self._size += 1

    def pop_front(self) -> Any:
        """Remove and return the front element.  O(1)."""
        if self.is_empty():
            raise IndexError("pop_front from empty list")
        node = self._sentinel.next
        self._sentinel.next = node.next
        if node.next is None:
            self._tail = None
        self._size -= 1
        return node.data

    def append(self, value: Any) -> None:
        """Insert at the back.  O(1) with tail pointer."""
        node = _SLLNode(value)
        if self._tail is None:
            self._sentinel.next = node
        else:
            self._tail.next = node
        self._tail = node
        self._size += 1

    # ── O(n) operations ───────────────────────────────────

    def search(self, value: Any) -> Optional[_SLLNode]:
        """Return the first node with node.data == value, or None.  O(n)."""
        cur = self._sentinel.next
        while cur is not None:
            if cur.data == value:
                return cur
            cur = cur.next
        return None

    def delete(self, value: Any) -> bool:
        """
        Remove the first occurrence of value.
        Returns True if found and removed, False otherwise.  O(n).
        """
        prev = self._sentinel
        cur  = self._sentinel.next
        while cur is not None:
            if cur.data == value:
                prev.next = cur.next
                if cur.next is None:          # removing the tail
                    self._tail = prev if prev is not self._sentinel else None
                self._size -= 1
                return True
            prev = cur
            cur  = cur.next
        return False

    def traverse(self) -> list:
        """Return a plain Python list of all values in order.  O(n)."""
        result = []
        cur = self._sentinel.next
        while cur is not None:
            result.append(cur.data)
            cur = cur.next
        return result

    def is_empty(self) -> bool:
        return self._size == 0

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return "SLL(" + " → ".join(map(str, self.traverse())) + ")"


# ════════════════════════════════════════════════════════
#  6.  ROOTED TREE  (left-child / right-sibling representation)
# ════════════════════════════════════════════════════════

class TreeNode:
    """
    Node in a rooted tree (CLRS §10.4).
    Uses the left-child / right-sibling scheme so that the number
    of children per node need not be fixed in advance.

    Attributes
    ----------
    data            : Any value stored at this node
    parent          : pointer to parent  (None for root)
    left_child      : pointer to first child
    right_sibling   : pointer to next sibling
    """

    __slots__ = ("data", "parent", "left_child", "right_sibling")

    def __init__(self, data: Any) -> None:
        self.data = data
        self.parent:        Optional[TreeNode] = None
        self.left_child:    Optional[TreeNode] = None
        self.right_sibling: Optional[TreeNode] = None


class RootedTree:
    """
    General-purpose rooted tree.

    Operation              | Complexity
    -----------------------|------------
    add_child              | O(k) where k = current number of children
    get_children           | O(k)
    depth (of a node)      | O(h) where h = tree height
    height (of subtree)    | O(n)
    preorder / postorder   | O(n)
    """

    def __init__(self, root_data: Any) -> None:
        self.root = TreeNode(root_data)

    def add_child(self, parent_node: TreeNode, child_data: Any) -> TreeNode:
        """
        Attach a new leaf as the last child of parent_node.
        Traverses the sibling chain to find the insertion point.  O(k).
        """
        child = TreeNode(child_data)
        child.parent = parent_node
        if parent_node.left_child is None:
            parent_node.left_child = child
        else:
            sib = parent_node.left_child
            while sib.right_sibling is not None:
                sib = sib.right_sibling
            sib.right_sibling = child
        return child

    def get_children(self, node: TreeNode) -> list:
        """Return an ordered list of node's children.  O(k)."""
        children = []
        cur = node.left_child
        while cur is not None:
            children.append(cur)
            cur = cur.right_sibling
        return children

    def depth(self, node: TreeNode) -> int:
        """Number of edges from root to node.  O(h)."""
        d = 0
        while node.parent is not None:
            d += 1
            node = node.parent
        return d

    def height(self, node: Optional[TreeNode] = None) -> int:
        """Maximum number of edges on any downward path from node.  O(n)."""
        if node is None:
            node = self.root
        if node.left_child is None:
            return 0
        return 1 + max(self.height(c) for c in self.get_children(node))

    def preorder(self, node: Optional[TreeNode] = None) -> list:
        """Visit root, then recursively visit children left-to-right.  O(n)."""
        if node is None:
            node = self.root
        result = [node.data]
        for child in self.get_children(node):
            result.extend(self.preorder(child))
        return result

    def postorder(self, node: Optional[TreeNode] = None) -> list:
        """Visit children, then root.  O(n)."""
        if node is None:
            node = self.root
        result = []
        for child in self.get_children(node):
            result.extend(self.postorder(child))
        result.append(node.data)
        return result


# ════════════════════════════════════════════════════════
#  7.  DEMONSTRATION / SMOKE TESTS
# ════════════════════════════════════════════════════════

def demo_dynamic_array():
    print("── DynamicArray ──────────────────────────")
    a = DynamicArray()
    for i in range(7):
        a.append(i * 10)
    print("After appending 0,10,...,60:", a)
    a.insert(2, 99)
    print("After insert(2, 99):", a)
    a.delete(2)
    print("After delete(2):", a)
    print()


def demo_matrix():
    print("── Matrix ────────────────────────────────")
    m = Matrix(3, 3)
    for r in range(3):
        for c in range(3):
            m.set(r, c, r * 3 + c + 1)
    print(m)
    m.add_row([10, 20, 30])
    print("After add_row([10,20,30]):")
    print(m)
    m.delete_row(0)
    print("After delete_row(0):")
    print(m)
    print()


def demo_stack():
    print("── Stack ─────────────────────────────────")
    s = Stack()
    for v in [1, 2, 3, 4, 5]:
        s.push(v)
    print("Pushed 1–5:", s)
    print("pop →", s.pop(), "| peek →", s.peek())
    print("Stack after pop:", s)
    print()


def demo_queue():
    print("── Queue ─────────────────────────────────")
    q = Queue()
    for v in ['a', 'b', 'c', 'd']:
        q.enqueue(v)
    print("Enqueued a,b,c,d:", q)
    print("dequeue →", q.dequeue())
    q.enqueue('e')
    print("After dequeue + enqueue 'e':", q)
    print()


def demo_linked_list():
    print("── SinglyLinkedList ──────────────────────")
    ll = SinglyLinkedList()
    ll.append(10)
    ll.append(20)
    ll.append(30)
    ll.push_front(5)
    print("append 10,20,30 then push_front 5:", ll)
    ll.delete(20)
    print("After delete(20):", ll)
    print("Search(10):", ll.search(10).data)
    print("pop_front →", ll.pop_front(), "| list:", ll)
    print()


def demo_rooted_tree():
    print("── RootedTree ────────────────────────────")
    t = RootedTree("A")
    b = t.add_child(t.root, "B")
    c = t.add_child(t.root, "C")
    d = t.add_child(t.root, "D")
    t.add_child(b, "E")
    t.add_child(b, "F")
    t.add_child(c, "G")
    print("Preorder:", t.preorder())
    print("Postorder:", t.postorder())
    print("Height:", t.height())
    print(f"Depth of G: {t.depth(t.get_children(c)[0])}")
    print()


if __name__ == "__main__":
    demo_dynamic_array()
    demo_matrix()
    demo_stack()
    demo_queue()
    demo_linked_list()
    demo_rooted_tree()
