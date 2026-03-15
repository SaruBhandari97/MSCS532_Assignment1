import csv
import random
import time
import tracemalloc



# Merge Sort Implementation

def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)


def merge(left, right):
    result = []
    i = 0
    j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result


# Quick Sort Implementation

def quick_sort(arr):
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)



# Dataset Generators

def sorted_data(n):
    return list(range(n))


def reverse_data(n):
    return list(range(n, 0, -1))


def random_data(n):
    return [random.randint(1, 100000) for _ in range(n)]


# Performance Measurement

def measure(sort_func, data):
    tracemalloc.start()
    start = time.perf_counter()

    sorted_result = sort_func(data.copy())

    end = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    execution_time = end - start
    peak_memory_kb = peak / 1024

    return execution_time, peak_memory_kb, sorted_result


# Save Results to CSV

def save_results_to_csv(results, filename="performance_results.csv"):
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Algorithm",
            "Dataset Type",
            "Input Size",
            "Execution Time (s)",
            "Peak Memory Usage (KB)"
        ])
        writer.writerows(results)



# Main Program

if __name__ == "__main__":
    sizes = [1000, 5000, 10000]

    datasets = {
        "Sorted": sorted_data,
        "Reverse Sorted": reverse_data,
        "Random": random_data
    }

    algorithms = {
        "Merge Sort": merge_sort,
        "Quick Sort": quick_sort
    }

    results = []

    print("=" * 90)
    print("Divide-and-Conquer Algorithm Performance Comparison")
    print("=" * 90)

    for size in sizes:
        print(f"\nDataset Size: {size}")
        print("-" * 90)

        for dataset_name, generator in datasets.items():
            data = generator(size)

            for algo_name, algo in algorithms.items():
                time_taken, memory_used, _ = measure(algo, data)

                results.append([
                    algo_name,
                    dataset_name,
                    size,
                    round(time_taken, 6),
                    round(memory_used, 2)
                ])

                print(
                    f"{algo_name:<12} | {dataset_name:<15} | "
                    f"Time: {time_taken:.6f} s | Memory: {memory_used:.2f} KB"
                )

    save_results_to_csv(results)

    print("\nResults have been saved to 'performance_results.csv'")