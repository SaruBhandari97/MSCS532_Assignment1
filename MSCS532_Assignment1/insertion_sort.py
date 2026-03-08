# MSCS532 Assignment 1
# Insertion Sort in monotonically decreasing order

def insertion_sort_desc(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1

        while j >= 0 and arr[j] < key:
            arr[j + 1] = arr[j]
            j -= 1

        arr[j + 1] = key

    return arr


numbers = [5, 2, 9, 1, 7, 6]

print("Original array:", numbers)
sorted_numbers = insertion_sort_desc(numbers)
print("Sorted array:", sorted_numbers)