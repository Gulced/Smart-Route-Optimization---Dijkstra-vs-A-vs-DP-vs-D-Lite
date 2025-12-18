import heapq
import time
import math

def heuristic_fn(a, b, mode="manhattan"):
    (r1, c1) = a
    (r2, c2) = b
    dr = abs(r1 - r2)
    dc = abs(c1 - c2)

    if mode == "manhattan":
        return dr + dc
    if mode == "euclidean":
        return math.sqrt(dr*dr + dc*dc)
    if mode == "chebyshev":
        return max(dr, dc)

    return dr + dc

def astar(grid, start, goal, heuristic="manhattan"):
    t0 = time.perf_counter()

    rows, cols = grid.shape
    INF = float("inf")

    g = {start: 0}
    prev = {}
    visited = set()
    expanded = 0

    pq = [(heuristic_fn(start, goal, heuristic), 0, start)]  # (f, g, node)

    while pq:
        f, gcur, u = heapq.heappop(pq)
        if u in visited:
            continue

        visited.add(u)
        expanded += 1

        if u == goal:
            break

        r, c = u
        for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
            nr, nc = r + dr, c + dc
            v = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                tentative = gcur + 1
                if tentative < g.get(v, INF):
                    g[v] = tentative
                    prev[v] = u
                    fv = tentative + heuristic_fn(v, goal, heuristic)
                    heapq.heappush(pq, (fv, tentative, v))

    path = []
    if goal in g:
        cur = goal
        while cur != start:
            path.append(cur)
            cur = prev[cur]
        path.append(start)
        path.reverse()

    runtime = time.perf_counter() - t0
    return path, g.get(goal, None), expanded, runtime, visited