import time
import numpy as np

DIRS = [(0,1), (1,0), (-1,0), (0,-1)]

def dp_shortest_path(grid, start, goal, max_iters=200):
    """
    Dynamic Programming tabanlı grid shortest path.

    grid: numpy array (0 free, 1 obstacle)
    start: (r,c)
    goal: (r,c)
    max_iters: DP güncelleme sayısı (bellman-like)

    Returns:
        path
        cost
        expanded_cells (DP update count)
        runtime
        visited_order (DP propagation sırasında dokunulan hücreler)
    """

    n = grid.shape[0]
    INF = 10**9

    # DP tablosu
    dp = np.full((n, n), INF, dtype=float)

    # Start 0 maliyet
    dp[start] = 0

    visited_order = []
    t0 = time.perf_counter()

    # Bellman-Ford'un grid versiyonu gibi çalışıyoruz
    for iteration in range(max_iters):

        changed = False

        for r in range(n):
            for c in range(n):
                if grid[r][c] == 1:
                    continue  # obstacle

                best = dp[r][c]

                for dr, dc in DIRS:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < n and 0 <= nc < n:
                        if grid[nr][nc] == 1:
                            continue
                        best = min(best, dp[nr][nc] + 1)

                if best < dp[r][c]:
                    dp[r][c] = best
                    changed = True
                    visited_order.append((r,c))

        if not changed:
            # Artık değişiklik yok → DP stabilize oldu
            break

    runtime = time.perf_counter() - t0
    expanded = len(visited_order)

    # Eğer goal erişilemezse
    if dp[goal] >= INF:
        return None, None, expanded, runtime, visited_order

    # Şimdi path çıkaralım (Goal → Start yönünde)
    path = []
    node = goal
    while node != start:
        path.append(node)
        r, c = node

        best = dp[r][c]
        next_node = None

        for dr, dc in DIRS:
            nr, nc = r+dr, c+dc
            if 0 <= nr < n and 0 <= nc < n:
                if dp[nr][nc] < best:
                    best = dp[nr][nc]
                    next_node = (nr, nc)

        if next_node is None:
            # DP tablo tutarsızlığı: yol yok
            return None, None, expanded, runtime, visited_order

        node = next_node

    path.append(start)
    path.reverse()

    return path, dp[goal], expanded, runtime, visited_order
