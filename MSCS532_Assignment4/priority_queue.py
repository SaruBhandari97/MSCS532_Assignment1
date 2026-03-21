"""
priority_queue.py
=================
Assignment 4 – Heap Data Structures: Implementation, Analysis, and Applications
Course:  Data Structures and Algorithms

Overview
--------
This module implements a max-priority queue backed by an array-based binary
max-heap.  It is designed to support a task scheduler simulation in which
tasks with higher priority values are executed before lower-priority ones.

Two classes are provided:
    Task                  -- a dataclass representing a schedulable unit of work
    MaxHeapPriorityQueue  -- the priority queue itself

Design Decisions
----------------
1. Array-based binary max-heap
   Chosen over a linked-tree implementation for three reasons:
     - No per-node pointer overhead (parent/child indices are computed
       arithmetically from the element's position).
     - Contiguous memory layout improves CPU cache utilization.
     - Python's list provides amortized O(1) append and O(1) index access,
       which are the only list operations the heap relies on.

2. Max-heap semantics (highest priority = highest urgency)
   The task with the greatest priority value always occupies index 0,
   enabling O(1) peek and O(log n) extract_max without any extra bookkeeping.
   This maps naturally to preemptive priority scheduling: the CPU always
   services the most urgent task first.

3. Auxiliary position map (_pos)
   A Python dict mapping task_id -> current heap index.
   Without this map, increase_key / decrease_key would require an O(n)
   linear scan to find the target task.  With it, the task is located in
   O(1) and the subsequent sift-up / sift-down costs O(log n).
   Every heap mutation (swap, insert, extract) updates _pos as an invariant.

4. Task comparison via priority only
   The Task dataclass overrides Python's dunder comparison methods to compare
   solely on the priority field.  This decouples scheduling order from
   incidental attributes like task_id or insertion timestamp.

Complexity Summary
------------------
Operation          Time        Space   Notes
-----------        -------     ------  -----------------------------------
insert()           O(log n)    O(1)    Append + sift up
extract_max()      O(log n)    O(1)    Swap root + sift down
increase_key()     O(log n)    O(1)    Update + sift up (moves toward root)
decrease_key()     O(log n)    O(1)    Update + sift down (moves toward leaf)
peek()             O(1)        O(1)    Return heap[0]
is_empty()         O(1)        O(1)    Check len(heap) == 0

Usage
-----
    from priority_queue import MaxHeapPriorityQueue, Task

    pq = MaxHeapPriorityQueue()
    pq.insert(Task(task_id=1, priority=5, description="Write tests"))
    pq.insert(Task(task_id=2, priority=9, description="Fix critical bug"))

    task = pq.extract_max()   # Returns Task(id=2, priority=9, ...)
    pq.increase_key(1, 10)    # Task 1 now has priority 10
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import time


# ==============================================================================
# SECTION 1 -- Task Data Class
# ==============================================================================

@dataclass(order=False)   # order=False because we define our own comparators
class Task:
    """
    Represents a single schedulable unit of work.

    Attributes
    ----------
    task_id      : int
        Unique integer identifier.  Used as the key in the position map.
    priority     : int
        Scheduling weight.  Higher values are processed first (max-heap).
        Valid range: any integer; ties are broken arbitrarily.
    arrival_time : float
        Simulation timestamp at which the task entered the system.
        Defaults to the current wall-clock time (time.time()).
    deadline     : Optional[float]
        Absolute deadline for task completion (seconds from epoch).
        None means no deadline constraint.
    description  : str
        Human-readable label for logging and display purposes.

    Comparison Semantics
    --------------------
    All six comparison operators are explicitly defined and compare ONLY the
    priority field.  This means two Task objects with equal priority are
    considered equal in ordering terms, even if their other fields differ.
    __eq__ is the exception: it checks task_id for identity, which ensures
    set/dict membership works correctly and prevents accidental deduplication
    of tasks with the same priority.
    """
    task_id:      int
    priority:     int
    arrival_time: float = field(default_factory=time.time)
    deadline:     Optional[float] = None
    description:  str  = ""

    # -- Priority-based ordering (drives heap comparisons) -------------------
    def __lt__(self, other: Task) -> bool:
        return self.priority < other.priority

    def __le__(self, other: Task) -> bool:
        return self.priority <= other.priority

    def __gt__(self, other: Task) -> bool:
        return self.priority > other.priority

    def __ge__(self, other: Task) -> bool:
        return self.priority >= other.priority

    def __eq__(self, other: object) -> bool:
        # Identity check by task_id so that set/dict operations work correctly.
        # Two tasks with the same priority but different IDs are NOT equal.
        return isinstance(other, Task) and self.task_id == other.task_id

    def __hash__(self) -> int:
        # Required whenever __eq__ is overridden: makes Task hashable for sets.
        return hash(self.task_id)

    def __repr__(self) -> str:
        return (f"Task(id={self.task_id}, priority={self.priority}, "
                f"desc='{self.description}')")


# ==============================================================================
# SECTION 2 -- Max-Heap Priority Queue
# ==============================================================================

class MaxHeapPriorityQueue:
    """
    A max-priority queue backed by an array-based binary max-heap.

    Internal Representation
    -----------------------
    _heap : list[Task]
        The heap array.  For any element at index i:
            parent      = (i - 1) // 2
            left child  = 2 * i + 1
            right child = 2 * i + 2
        The max-heap invariant guarantees: _heap[parent].priority
        >= _heap[child].priority for all valid indices.

    _pos : dict[int, int]
        Maps task_id -> current index in _heap.  Updated atomically with
        every swap so the map is always consistent with _heap layout.
        Enables O(1) task lookup for increase_key / decrease_key.

    Invariant
    ---------
    After every public method returns, the following holds:
        1. _heap satisfies the max-heap property.
        2. _pos[t.task_id] == i  <=>  _heap[i] == t  for all tasks t.
    """

    def __init__(self) -> None:
        self._heap: list[Task] = []       # The backing heap array
        self._pos:  dict[int, int] = {}   # Position map: task_id -> heap index

    # ==========================================================================
    # Private Helper Methods
    # ==========================================================================

    def _swap(self, i: int, j: int) -> None:
        """
        Swap the elements at indices i and j in _heap, and update _pos.

        Keeping _pos consistent on every swap is critical.  If we swapped
        _heap without updating _pos, the position map would point to stale
        indices and future increase_key / decrease_key calls would corrupt
        the heap.

        Time Complexity : O(1)
        """
        # Update the position map BEFORE moving the elements, while we still
        # have a reference to which task_id is at each index.
        self._pos[self._heap[i].task_id] = j
        self._pos[self._heap[j].task_id] = i

        # Perform the actual swap in the heap array
        self._heap[i], self._heap[j] = self._heap[j], self._heap[i]

    def _sift_up(self, i: int) -> None:
        """
        Move the element at index i upward until the heap property is restored.

        Called after insert() or increase_key(), when an element's priority
        may be higher than its parent's.  The element "bubbles up" the tree
        by repeatedly swapping with its parent until either:
            (a) it reaches the root (i == 0), or
            (b) its priority is <= its parent's priority.

        Time Complexity : O(log n) -- traverses at most one root-to-node path.
        """
        while i > 0:
            parent = (i - 1) // 2   # Parent of node i

            # If the current node outranks its parent, swap upward
            if self._heap[i] > self._heap[parent]:
                self._swap(i, parent)
                i = parent           # Continue from the parent's position
            else:
                break                # Heap property satisfied; stop early

    def _sift_down(self, i: int) -> None:
        """
        Move the element at index i downward until the heap property is restored.

        Called after extract_max() or decrease_key(), when an element's
        priority may be lower than one or both of its children.  The element
        "sinks" down the tree by repeatedly swapping with the larger child
        until either:
            (a) it reaches a leaf (no children), or
            (b) it is >= both children.

        Time Complexity : O(log n) -- traverses at most one root-to-leaf path.
        """
        n = len(self._heap)

        while True:
            largest = i               # Assume current node is the largest
            left    = 2 * i + 1       # Left child index
            right   = 2 * i + 2       # Right child index

            # Check if left child exists and outranks the current node
            if left < n and self._heap[left] > self._heap[largest]:
                largest = left

            # Check if right child exists and outranks the current largest
            if right < n and self._heap[right] > self._heap[largest]:
                largest = right

            if largest != i:
                # A child is larger -- swap downward and continue from there
                self._swap(i, largest)
                i = largest
            else:
                break    # Heap property satisfied; stop

    # ==========================================================================
    # Public API
    # ==========================================================================

    def insert(self, task: Task) -> None:
        """
        Insert a new task into the priority queue.

        Process
        -------
        1. Append the task to the end of the heap array.      O(1) amortised
        2. Record its index in the position map.               O(1)
        3. Sift up to restore the max-heap property.          O(log n)

        The append + sift-up strategy is correct because appending at the end
        maintains the complete binary tree shape invariant, and then sift-up
        corrects any priority violation between the new leaf and its ancestors.

        Parameters
        ----------
        task : Task
            The task to insert.  Its task_id must be unique within this queue.

        Raises
        ------
        ValueError
            If a task with the same task_id is already present.

        Time Complexity : O(log n)
        """
        # Guard against duplicate task IDs to keep _pos consistent
        if task.task_id in self._pos:
            raise ValueError(f"Task {task.task_id} already exists in queue.")

        idx = len(self._heap)       # New element will be appended at this index
        self._heap.append(task)     # Add task to the end of the heap
        self._pos[task.task_id] = idx  # Record its initial position

        # The new element may violate the heap property with its parent --
        # sift it upward until the invariant is restored.
        self._sift_up(idx)

    def extract_max(self) -> Task:
        """
        Remove and return the task with the highest priority.

        Process
        -------
        1. Swap the root (maximum) with the last element.     O(1)
        2. Pop the last element (the old root/maximum).       O(1)
        3. Remove the popped task from the position map.      O(1)
        4. Sift down the new root to restore heap order.      O(log n)

        Why swap before popping?
        Directly removing the root would leave a gap at index 0.  By first
        swapping with the last element, we maintain the complete binary tree
        shape (still all levels full except possibly the last), and only need
        to fix the priority violation at the root by sifting down.

        Returns
        -------
        Task
            The highest-priority task, removed from the queue.

        Raises
        ------
        IndexError
            If the queue is empty.

        Time Complexity : O(log n)
        """
        if self.is_empty():
            raise IndexError("extract_max() called on an empty priority queue.")

        # Move the last element to the root position (maintains shape invariant)
        self._swap(0, len(self._heap) - 1)

        # Pop the last element (which is the old maximum after the swap)
        task = self._heap.pop()
        del self._pos[task.task_id]   # Remove from position map

        # The new root may violate the heap property with its children --
        # sift it downward until the invariant is restored.
        if self._heap:
            self._sift_down(0)

        return task

    def increase_key(self, task_id: int, new_priority: int) -> None:
        """
        Increase the priority of an existing task.

        A higher priority moves a task closer to the root, so we restore the
        heap property by sifting the task upward after updating its priority.

        Process
        -------
        1. Look up the task's current index via the position map.   O(1)
        2. Validate that new_priority strictly exceeds the current. O(1)
        3. Update the priority field in place.                      O(1)
        4. Sift up to restore the max-heap property.               O(log n)

        Parameters
        ----------
        task_id      : int
            Identifier of the task whose priority is being raised.
        new_priority : int
            The new, higher priority value.

        Raises
        ------
        KeyError
            If no task with task_id exists in the queue.
        ValueError
            If new_priority does not strictly exceed the current priority.

        Time Complexity : O(log n)
        """
        if task_id not in self._pos:
            raise KeyError(f"Task {task_id} not found in priority queue.")

        idx = self._pos[task_id]    # O(1) lookup via position map

        if new_priority <= self._heap[idx].priority:
            raise ValueError(
                f"increase_key requires new_priority ({new_priority}) > "
                f"current priority ({self._heap[idx].priority})."
            )

        self._heap[idx].priority = new_priority   # Update priority in place

        # Higher priority may now violate the heap property with the parent --
        # sift upward to restore it.
        self._sift_up(idx)

    def decrease_key(self, task_id: int, new_priority: int) -> None:
        """
        Decrease the priority of an existing task.

        A lower priority moves a task away from the root, so we restore the
        heap property by sifting the task downward after updating its priority.

        Process
        -------
        1. Look up the task's current index via the position map.   O(1)
        2. Validate that new_priority strictly precedes the current. O(1)
        3. Update the priority field in place.                      O(1)
        4. Sift down to restore the max-heap property.             O(log n)

        Parameters
        ----------
        task_id      : int
            Identifier of the task whose priority is being lowered.
        new_priority : int
            The new, lower priority value.

        Raises
        ------
        KeyError
            If no task with task_id exists in the queue.
        ValueError
            If new_priority does not strictly precede the current priority.

        Time Complexity : O(log n)
        """
        if task_id not in self._pos:
            raise KeyError(f"Task {task_id} not found in priority queue.")

        idx = self._pos[task_id]    # O(1) lookup via position map

        if new_priority >= self._heap[idx].priority:
            raise ValueError(
                f"decrease_key requires new_priority ({new_priority}) < "
                f"current priority ({self._heap[idx].priority})."
            )

        self._heap[idx].priority = new_priority   # Update priority in place

        # Lower priority may now violate the heap property with the children --
        # sift downward to restore it.
        self._sift_down(idx)

    def peek(self) -> Task:
        """
        Return the highest-priority task without removing it.

        The max-heap invariant guarantees the maximum is always at index 0,
        so this is a simple array access.

        Returns
        -------
        Task
            The highest-priority task currently in the queue.

        Raises
        ------
        IndexError
            If the queue is empty.

        Time Complexity : O(1)
        """
        if self.is_empty():
            raise IndexError("peek() called on an empty priority queue.")
        return self._heap[0]

    def is_empty(self) -> bool:
        """
        Return True if the priority queue contains no tasks.

        Time Complexity : O(1)
        """
        return len(self._heap) == 0

    def __len__(self) -> int:
        """Return the number of tasks currently in the queue."""
        return len(self._heap)

    def __repr__(self) -> str:
        top = self._heap[0] if self._heap else None
        return f"MaxHeapPriorityQueue(size={len(self)}, top={top})"


# ==============================================================================
# SECTION 3 -- Scheduler Simulation / Demo
# ==============================================================================

def run_scheduler_demo() -> list:
    """
    Demonstrate the priority queue through a realistic task scheduling scenario.

    Scenario
    --------
    Six tasks representing a realistic workday workload are inserted with
    varying priorities.  The demo then exercises increase_key and decrease_key
    to simulate a supervisor re-prioritizing tasks mid-session.  Finally,
    all tasks are drained in priority order to show the execution sequence.

    This function prints a step-by-step log of every operation and returns
    the ordered list of executed Task objects for downstream use.

    Returns
    -------
    list[Task]
        Tasks in the order they were extracted (highest to lowest priority
        after all key modifications).
    """
    pq = MaxHeapPriorityQueue()

    # -- Step 1: Insert an initial batch of tasks ----------------------------
    tasks = [
        Task(task_id=1, priority=5,  description="Send weekly report"),
        Task(task_id=2, priority=10, description="Fix critical production bug"),
        Task(task_id=3, priority=3,  description="Update documentation"),
        Task(task_id=4, priority=7,  description="Code review"),
        Task(task_id=5, priority=1,  description="Team lunch reservation"),
        Task(task_id=6, priority=9,  description="Security patch deployment"),
    ]

    print("=" * 60)
    print("  Task Scheduler Simulation")
    print("=" * 60)
    print("\n-- Phase 1: Inserting tasks --")
    for t in tasks:
        pq.insert(t)
        print(f"  Inserted  {t}")

    print(f"\n  Queue size : {len(pq)}")
    print(f"  Highest priority task: {pq.peek()}")

    # -- Step 2: Simulate a supervisor re-prioritizing two tasks -------------
    print("\n-- Phase 2: Modifying priorities --")

    # Task 3 (documentation) is now needed urgently before a product launch
    print("  Increasing task 3 priority: 3 -> 8 (doc update now urgent)")
    pq.increase_key(3, 8)

    # Task 6 (security patch) is deferred to next sprint
    print("  Decreasing task 6 priority: 9 -> 4 (security patch deferred)")
    pq.decrease_key(6, 4)

    print(f"\n  Highest priority task after changes: {pq.peek()}")

    # -- Step 3: Execute all tasks in priority order -------------------------
    print("\n-- Phase 3: Executing tasks in priority order --")
    execution_order = []
    while not pq.is_empty():
        t = pq.extract_max()
        execution_order.append(t)
        print(f"  Executed: {t}")

    print("\n  All tasks completed.")
    return execution_order


# ==============================================================================
# SECTION 4 -- Script Entry Point
# ==============================================================================

if __name__ == "__main__":
    run_scheduler_demo()
