import random
import matplotlib.pyplot as plt
import numpy as np


class GridMap:
    def __init__(self, n, obstacle_rate=0.25):
        self.n = n
        self.obstacle_rate = obstacle_rate
        self.grid = None
        self.start = (0, 0)
        self.goal = (n - 1, n - 1)

    def generate(self):
        """n×n grid üret, obstacle_rate’e göre engel ekle."""
        self.grid = np.zeros((self.n, self.n), dtype=int)

        for i in range(self.n):
            for j in range(self.n):
                if random.random() < self.obstacle_rate:
                    self.grid[i][j] = 1  # 1 = obstacle

        # Start ve goal daima açık olsun
        self.grid[self.start] = 0
        self.grid[self.goal] = 0

    def show(self):
        """Gridi matplotlib ile basitçe gösterir."""
        if self.grid is None:
            raise ValueError("Grid map not generated yet.")

        plt.figure(figsize=(6, 6))
        plt.imshow(self.grid, cmap="binary")
        plt.title(f"Grid Map ({self.n}x{self.n})")

        # start = yeşil, goal = kırmızı
        plt.scatter(self.start[1], self.start[0], color='green', s=80, label='Start')
        plt.scatter(self.goal[1], self.goal[0], color='red', s=80, label='Goal')

        plt.legend()
        plt.gca().invert_yaxis()
        plt.show()
