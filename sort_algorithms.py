import time

def merge_sort(arr, key):
    ops = 0
    t0 = time.perf_counter()

    def _merge_sort(a):
        nonlocal ops
        if len(a) <= 1:
            return a

        mid = len(a) // 2
        left = _merge_sort(a[:mid])
        right = _merge_sort(a[mid:])

        return _merge(left, right)

    def _merge(left, right):
        nonlocal ops
        result = []
        i = j = 0

        while i < len(left) and j < len(right):
            ops += 1
            if left[i][key] <= right[j][key]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1

        result.extend(left[i:])
        result.extend(right[j:])
        return result

    sorted_arr = _merge_sort(arr)
    runtime = (time.perf_counter() - t0) * 1000

    return sorted_arr, runtime, ops


def quick_sort(arr, key):
    ops = 0
    t0 = time.perf_counter()

    def _quick_sort(a):
        nonlocal ops
        if len(a) <= 1:
            return a

        pivot = a[len(a) // 2][key]
        left, mid, right = [], [], []

        for x in a:
            ops += 1
            if x[key] < pivot:
                left.append(x)
            elif x[key] > pivot:
                right.append(x)
            else:
                mid.append(x)

        return _quick_sort(left) + mid + _quick_sort(right)

    sorted_arr = _quick_sort(arr)
    runtime = (time.perf_counter() - t0) * 1000

    return sorted_arr, runtime, ops
