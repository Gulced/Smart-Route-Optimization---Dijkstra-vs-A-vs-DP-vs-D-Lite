import heapq
import time
import math

DIRS = [(0,1),(1,0),(-1,0),(0,-1)]

INF = 10**9

def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])  # manhattan

class DStarLite:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.n = grid.shape[0]

        self.start = start
        self.goal = goal

        self.g = {}
        self.rhs = {}

        self.U = []  # priority queue

        # Varsayılan değerler
        for r in range(self.n):
            for c in range(self.n):
                self.g[(r,c)] = INF
                self.rhs[(r,c)] = INF

        self.rhs[self.goal] = 0  # hedef için rhs = 0

        self.km = 0  # key modifier

        self._insert(self.goal, self._calculate_key(self.goal))

    # ------------------------
    # PRIORITY QUEUE HELPERS
    # ------------------------

    def _calculate_key(self, node):
        g_rhs = min(self.g[node], self.rhs[node])
        return (g_rhs + heuristic(self.start, node) + self.km, g_rhs)

    def _insert(self, node, key):
        heapq.heappush(self.U, (key, node))

    def _update_vertex(self, node):
        if node != self.goal:
            min_rhs = INF
            for nr, nc in self._neighbors(node):
                min_rhs = min(min_rhs, self.g[(nr, nc)] + 1)
            self.rhs[node] = min_rhs

        # Eğer g ≠ rhs ise queue’ya ekle
        if self.g[node] != self.rhs[node]:
            self._insert(node, self._calculate_key(node))

    def _neighbors(self, node):
        (r, c) = node
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.n and 0 <= nc < self.n:
                if self.grid[nr][nc] == 0:  # free cell
                    yield (nr, nc)

    # ------------------------
    # MAIN COMPUTE
    # ------------------------

    def _compute_shortest_path(self):
        expanded = 0

        while self.U:
            (k_old, node) = heapq.heappop(self.U)
            k_new = self._calculate_key(node)

            # Eğer eski entry ise atla
            if k_old < k_new:
                self._insert(node, k_new)
                continue

            expanded += 1

            # Case 1: g > rhs → improve g
            if self.g[node] > self.rhs[node]:
                self.g[node] = self.rhs[node]

                # Komşuları güncelle
                for nr, nc in self._neighbors(node):
                    self._update_vertex((nr, nc))

            # Case 2: g < rhs → degrade
            else:
                self.g[node] = INF
                for nr, nc in self._neighbors(node):
                    self._update_vertex((nr, nc))
                self._update_vertex(node)

            # Eğer start çözümlendiyse durabilir
            if self.g[self.start] == self.rhs[self.start]:
                break

        return expanded

    # ------------------------
    # PUBLIC METHOD: FIND PATH
    # ------------------------

    def find_path(self):
        t0 = time.perf_counter()

        expanded = self._compute_shortest_path()

        # PATH EXTRACTION
        if self.g[self.start] >= INF:
            runtime = time.perf_counter() - t0
            return None, None, expanded, runtime, 0

        path = []
        node = self.start
        update_count = 0

        while node != self.goal:
            path.append(node)
            best = INF
            next_node = None

            for nb in self._neighbors(node):
                val = self.g[nb]
                if val < best:
                    best = val
                    next_node = nb

            if next_node is None:
                runtime = time.perf_counter() - t0
                return None, None, expanded, runtime, update_count

            node = next_node

        path.append(self.goal)

        runtime = time.perf_counter() - t0
        return path, self.g[self.start], expanded, runtime, update_count

    # ------------------------
    # DYNAMIC UPDATE (Traffic change)
    # ------------------------

    def update_cell(self, cell, new_state):
        """
        Dinamik engel/değişiklik:
            new_state = 0 → boş
            new_state = 1 → engel
        """
        (r, c) = cell
        old = self.grid[r][c]
        self.grid[r][c] = new_state

        # Eğer durum değişmemişse bir şey yapma
        if old == new_state:
            return 0

        self.km += heuristic(self.start, self.goal)

        # Etkilenen node'ları güncelle
        for nb in self._neighbors(cell):
            self._update_vertex(nb)
        self._update_vertex(cell)

        # Yeniden planlama
        expanded = self._compute_shortest_path()
        return expanded
