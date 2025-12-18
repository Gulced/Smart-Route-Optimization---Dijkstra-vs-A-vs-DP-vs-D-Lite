import heapq
import time

def dijkstra(grid, start, goal):
    t0 = time.perf_counter()

    rows, cols = grid.shape
    INF = float("inf")

    dist = {start: 0}
    prev = {}
    visited = set()
    expanded = 0

    pq = [(0, start)]  # (cost, node)

    while pq:
        cost, u = heapq.heappop(pq)
        if u in visited:
            continue

        visited.add(u)
        expanded += 1

        if u == goal:
            break

        r, c = u
        for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                new_cost = cost + 1
                if new_cost < dist.get((nr, nc), INF):
                    dist[(nr, nc)] = new_cost
                    prev[(nr, nc)] = u
                    heapq.heappush(pq, (new_cost, (nr, nc)))

    path = []
    if goal in dist:
        cur = goal
        while cur != start:
            path.append(cur)
            cur = prev[cur]
        path.append(start)
        path.reverse()

    runtime = time.perf_counter() - t0
    return path, dist.get(goal, None), expanded, runtime, visited