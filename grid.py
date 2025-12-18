import numpy as np

class GridMap:
    """
    0 = boş, 1 = engel
    start = (0,0), goal = (n-1,n-1)
    """
    def __init__(self, n: int, obstacle_ratio: float, seed: int = 42):
        self.n = int(n)
        self.obstacle_ratio = float(obstacle_ratio)
        self.seed = int(seed)
        self.grid = None
        self.start = (0, 0)
        self.goal = (self.n - 1, self.n - 1)

    def generate(self):
        rng = np.random.default_rng(self.seed)
        n = self.n

        self.grid = (rng.random((n, n)) < self.obstacle_ratio).astype(int)

        # start/goal açık olsun
        self.start = (0, 0)
        self.goal = (n - 1, n - 1)
        self.grid[self.start] = 0
        self.grid[self.goal] = 0

        # start/goal çevresini de aç (pratikte yol bulmayı kolaylaştırır)
        for (r, c) in [self.start, self.goal]:
            for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n:
                    self.grid[rr, cc] = 0

        return self.grid