import heapq
import time

# 4 yönlü hareket (grid tabanlı)
DIRS = [(0,1), (1,0), (-1,0), (0,-1)]

def dijkstra(grid, start, goal):
    """
    grid: numpy array (0 = free, 1 = obstacle)
    start: (row, col)
    goal : (row, col)

    Returns:
        path           → bulunan yol (liste)
        cost           → adım sayısı
        expanded_nodes → ziyaret edilen düğüm sayısı
        runtime        → saniye cinsinden süre
        visited_order  → görselleştirme için tüm visited nodes
    """
    n = grid.shape[0]

    t0 = time.perf_counter()

    pq = []
    heapq.heappush(pq, (0, start))

    dist = {start: 0}
    parent = {start: None}

    visited_order = []
    expanded_nodes = 0

    while pq:
        cost, current = heapq.heappop(pq)

        if cost > dist[current]:
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

                new_cost = cost + 1

                if (nr, nc) not in dist or new_cost < dist[(nr, nc)]:
                    dist[(nr, nc)] = new_cost
                    parent[(nr, nc)] = current
                    heapq.heappush(pq, (new_cost, (nr, nc)))

    # Eğer hedefe ulaşılamadıysa
    if goal not in dist:
        return None, None, expanded_nodes, time.perf_counter() - t0, visited_order

    # Yolun geri izlenmesi
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]

    path.reverse()

    runtime = time.perf_counter() - t0
    return path, dist[goal], expanded_nodes, runtime, visited_order
