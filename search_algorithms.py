def linear_search(arr, key, threshold):
    """
    Linear Search:
    threshold değerinden küçük/eşit olan kayıtları bulur
    """
    results = []
    for item in arr:
        if item[key] <= threshold:
            results.append(item)
    return results


def binary_search(arr, key, threshold):
    """
    Binary Search:
    - arr sıralı OLMALIDIR
    - threshold <= olan ilk indeksi bulur
    """
    low = 0
    high = len(arr) - 1
    idx = -1

    while low <= high:
        mid = (low + high) // 2
        if arr[mid][key] <= threshold:
            idx = mid
            high = mid - 1
        else:
            low = mid + 1

    if idx == -1:
        return []

    return arr[:idx + 1]
