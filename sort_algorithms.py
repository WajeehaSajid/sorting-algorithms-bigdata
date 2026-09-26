"""
Part C — Sorting algorithm implementations from scratch.

All three functions:
- take a Python list `arr` (numeric OR string elements — comparisons
  work the same way for both in Python, so no special-casing is needed)
- sort ascending, in-place, and also return the list (convenient for chaining)
- use only comparisons and swaps — no built-in sort()/sorted()
"""


def bubble_sort(arr):
    """
    Bubble Sort.

    Repeatedly steps through the list, compares adjacent elements, and
    swaps them if they are in the wrong order. Each full pass "bubbles"
    the largest remaining element to its correct position at the end.

    Time complexity:
        Best case:    O(n)    -> already sorted input, with early-exit flag
        Average case: O(n^2)
        Worst case:   O(n^2)  -> reverse-sorted input
    Space complexity: O(1) (in-place)
    Stable: Yes (equal elements never swap past each other)
    """
    n = len(arr)
    for i in range(n - 1):
        swapped = False
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:      # already sorted -> stop early (gives O(n) best case)
            break
    return arr


def selection_sort(arr):
    """
    Selection Sort.

    For each position i, scans the unsorted remainder of the list to find
    the minimum element, then swaps it into position i.

    Time complexity:
        Best case:    O(n^2)  -> always scans the full remainder, regardless of order
        Average case: O(n^2)
        Worst case:   O(n^2)
    Space complexity: O(1) (in-place)
    Stable: No (the swap can move an equal element past another equal one)
    """
    n = len(arr)
    for i in range(n - 1):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr


def insertion_sort(arr):
    """
    Insertion Sort.

    Builds the sorted list one element at a time: takes the next element
    and shifts it leftward past every already-sorted element bigger than it.

    Time complexity:
        Best case:    O(n)    -> already sorted input, inner loop never shifts
        Average case: O(n^2)
        Worst case:   O(n^2)  -> reverse-sorted input, every element shifts to the front
    Space complexity: O(1) (in-place)
    Stable: Yes (an element is only shifted past strictly greater elements)
    """
    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


if __name__ == "__main__":
    # quick correctness self-test on numeric and string data
    import random

    for label, sample in [
        ("numeric", [random.randint(-100, 100) for _ in range(200)]),
        ("string", [f"CARRIER_{random.randint(0, 50)}" for _ in range(200)]),
    ]:
        for name, fn in [("bubble", bubble_sort), ("selection", selection_sort),
                          ("insertion", insertion_sort)]:
            got = fn(list(sample))
            assert got == sorted(sample), f"{name} FAILED on {label} data"
            print(f"{name:10s} sort correct on {label} data")
