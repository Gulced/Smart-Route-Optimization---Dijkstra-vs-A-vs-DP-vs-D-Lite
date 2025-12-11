import heapq
import time
import math

# Dijkstra ile aynı yönler:
DIRS = [(0, 1), (1, 0), (-1, 0), (0, -1)]

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

HEURISTICS = {
    "manhattan": manhattan,
    "euclidean": euclidean,
    "chebyshev": chebyshev
}

def astar(grid, start, goal, heuristic="manhattan"):
    """
    A* pathfinding on a grid.

    grid: numpy array (0 = free, 1 = obstacle)
    start: (row, col)
    goal : (row, col)
    heuristic: 'manhattan' / 'euclidean' / 'chebyshev'

    Returns:
        path           → bulunan yol (liste: [(r,c), ...])
        cost           → yol maliyeti (adım sayısı)
        expanded_nodes → genişletilen düğüm sayısı
        runtime        → saniye cinsinden süre
        visited_order  → sırayla ziyaret edilen düğümler (görselleştirme için)
    """
    if heuristic not in HEURISTICS:
        raise ValueError(f"Unknown heuristic: {heuristic}")

    h = HEURISTICS[heuristic]
    n = grid.shape[0]

    t0 = time.perf_counter()

    # f = g + h
    open_pq = []
    g_cost = {start: 0.0}
    f_cost = {start: h(start, goal)}
    parent = {start: None}

    heapq.heappush(open_pq, (f_cost[start], start))

    visited_order = []
    expanded_nodes = 0

    while open_pq:
        current_f, current = heapq.heappop(open_pq)

        # Eğer bu entry eskiyse atla
        if current_f != f_cost.get(current, float("inf")):
            continue

        visited_order.append(current)
        expanded_nodes += 1

        if current == goal:
            break

        r, c = current

        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:
                if grid[nr][nc] == 1:
                    continue  # obstacle

                tentative_g = g_cost[current] + 1  # birim maliyetli grid

                if tentative_g < g_cost.get((nr, nc), float("inf")):
                    g_cost[(nr, nc)] = tentative_g
                    parent[(nr, nc)] = current
                    f_val = tentative_g + h((nr, nc), goal)
                    f_cost[(nr, nc)] = f_val
                    heapq.heappush(open_pq, (f_val, (nr, nc)))

    # Eğer hedefe ulaşılmadıysa:
    if goal not in g_cost:
        runtime = time.perf_counter() - t0
        return None, None, expanded_nodes, runtime, visited_order

    # Yolun geri izlenmesi
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()

    runtime = time.perf_counter() - t0
    cost = g_cost[goal]

    return path, cost, expanded_nodes, runtime, visited_order
