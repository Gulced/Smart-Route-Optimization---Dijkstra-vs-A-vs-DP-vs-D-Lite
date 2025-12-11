import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
# Tkinter ile kullanılacak doğru backend
matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

from grid import GridMap


class RoutePlannerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Route Optimization - Dijkstra vs A* vs DP vs D* Lite")
        self.root.geometry("1250x720")

        # Data holders
        self.gridmap = None
        self.current_path = None
        self.dstar_planner = None  # D* Lite instance

        self._build_ui()

        # Pencere açılır açılmaz bir şey çizsin (boş ekran olmasın)
        self.draw_grid()

    # -------------------------------------------------------
    # UI BUILDER
    # -------------------------------------------------------
    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding=6)
        main_frame.pack(fill="both", expand=True)

        # LEFT PANEL (controls)
        left = ttk.Frame(main_frame)
        left.pack(side="left", fill="y")

        # Grid parameters
        lf_grid = ttk.LabelFrame(left, text="Grid Settings", padding=6)
        lf_grid.pack(fill="x", pady=4)

        ttk.Label(lf_grid, text="Grid Size (n):").pack(anchor="w")
        self.n_var = tk.IntVar(value=20)
        ttk.Entry(lf_grid, textvariable=self.n_var, width=10).pack(anchor="w")

        ttk.Label(lf_grid, text="Obstacle Rate (0-0.6):").pack(anchor="w", pady=(4, 0))
        self.obs_var = tk.DoubleVar(value=0.20)
        ttk.Entry(lf_grid, textvariable=self.obs_var, width=10).pack(anchor="w")

        ttk.Button(lf_grid, text="Generate Grid", command=self.generate_grid).pack(pady=6)

        # Algorithm selection
        lf_algo = ttk.LabelFrame(left, text="Algorithm", padding=6)
        lf_algo.pack(fill="x", pady=6)

        self.algo_var = tk.StringVar(value="dijkstra")

        ttk.Radiobutton(lf_algo, text="Dijkstra", variable=self.algo_var, value="dijkstra").pack(anchor="w")
        ttk.Radiobutton(lf_algo, text="A*", variable=self.algo_var, value="astar").pack(anchor="w")
        ttk.Radiobutton(lf_algo, text="DP", variable=self.algo_var, value="dp").pack(anchor="w")
        ttk.Radiobutton(lf_algo, text="Dynamic A*", variable=self.algo_var, value="dstar").pack(anchor="w")

        # Heuristic selection (only for A*)
        ttk.Label(lf_algo, text="Heuristic (A*):").pack(anchor="w", pady=(8, 0))
        self.heuristic_var = tk.StringVar(value="manhattan")
        heur_cb = ttk.Combobox(
            lf_algo,
            textvariable=self.heuristic_var,
            values=["manhattan", "euclidean", "chebyshev"],
            state="readonly"
        )
        heur_cb.pack(anchor="w")

        # Run buttons
        lf_run = ttk.LabelFrame(left, text="Actions", padding=6)
        lf_run.pack(fill="x", pady=6)

        ttk.Button(lf_run, text="Run Algorithm", command=self.run_algorithm).pack(fill="x", pady=3)
        ttk.Button(lf_run, text="Show Path", command=self.show_path).pack(fill="x", pady=3)
        ttk.Button(lf_run, text="Dynamic Update (D*)", command=self.dynamic_update).pack(fill="x", pady=3)
        ttk.Button(lf_run, text="Compare Results", command=self.open_compare_window).pack(fill="x", pady=3)

        # Results table
        lf_table = ttk.LabelFrame(left, text="Results (Runs)", padding=6)
        lf_table.pack(fill="both", expand=True, pady=6)

        cols = ("algo", "time", "expanded", "cost")
        self.table = ttk.Treeview(lf_table, columns=cols, show="headings", height=10)

        for c in cols:
            self.table.heading(c, text=c)
            self.table.column(c, anchor="center", width=90)

        self.table.pack(fill="both", expand=True)

        # RIGHT PANEL (matplotlib canvas)
        right = ttk.Frame(main_frame)
        right.pack(side="left", fill="both", expand=True)

        self.fig = plt.Figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # -------------------------------------------------------
    # ACTIONS
    # -------------------------------------------------------

    def generate_grid(self):
        n = self.n_var.get()
        obs = self.obs_var.get()

        if not (0 <= obs <= 0.6):
            messagebox.showerror("Error", "Obstacle rate must be between 0 and 0.6")
            return

        self.gridmap = GridMap(n, obs)
        self.gridmap.generate()

        self.dstar_planner = None
        self.current_path = None
        self.draw_grid()

    def run_algorithm(self):
        if self.gridmap is None:
            messagebox.showerror("Error", "Generate grid first.")
            return

        algo = self.algo_var.get()
        grid = self.gridmap.grid
        start = self.gridmap.start
        goal = self.gridmap.goal

        try:
            if algo == "dijkstra":
                from dijkstra import dijkstra
                path, cost, expanded, runtime, visited = dijkstra(grid, start, goal)

            elif algo == "astar":
                from astar import astar
                heuristic = self.heuristic_var.get()
                path, cost, expanded, runtime, visited = astar(grid, start, goal, heuristic)

            elif algo == "dp":
                from dp_path import dp_shortest_path
                path, cost, expanded, runtime, visited = dp_shortest_path(grid, start, goal)

            elif algo == "dstar":
                from dstar_lite import DStarLite
                if self.dstar_planner is None:
                    self.dstar_planner = DStarLite(grid, start, goal)
                path, cost, expanded, runtime, updates = self.dstar_planner.find_path()
            else:
                raise ValueError("Unknown algorithm")

        except Exception as e:
            messagebox.showerror("Error", f"Algo error: {e}")
            return

        if path is None:
            messagebox.showinfo("No Path", f"{algo} could not find a path.")
            return

        self.current_path = path
        self.draw_grid(path=path)

        time_ms = round(runtime * 1000, 3)
        self.table.insert("", "end", values=(algo.upper(), f"{time_ms} ms", expanded, cost))

    def show_path(self):
        if self.current_path is None:
            messagebox.showinfo("Info", "No path computed yet.")
        else:
            self.draw_grid(path=self.current_path)

    def dynamic_update(self):
        if self.algo_var.get() != "dstar":
            messagebox.showinfo("Info", "Dynamic update only works with D*")
            return

        if self.dstar_planner is None:
            messagebox.showerror("Error", "Run Dynamic A* first.")
            return

        n = self.gridmap.grid.shape[0]
        r = np.random.randint(0, n)
        c = np.random.randint(0, n)

        old = self.gridmap.grid[r][c]
        new = 1 - old
        self.gridmap.grid[r][c] = new

        expanded = self.dstar_planner.update_cell((r, c), new)
        path, cost, expanded2, runtime, _ = self.dstar_planner.find_path()

        if path:
            self.current_path = path
            self.draw_grid(path=path)

        messagebox.showinfo("Dynamic Update", f"Cell ({r},{c}) changed {old}->{new}")

    # -------------------------------------------------------
    # COMPARISON WINDOW
    # -------------------------------------------------------

    def open_compare_window(self):
        if not self.table.get_children():
            messagebox.showinfo("No Data", "Run at least one algorithm first.")
            return

        win = tk.Toplevel(self.root)
        win.title("Algorithm Comparison")
        win.geometry("1100x800")

        algos, times, expands, costs = [], [], [], []

        for row_id in self.table.get_children():
            algo, t, exp, cost = self.table.item(row_id)["values"]
            algos.append(algo)
            times.append(float(str(t).split()[0]))
            expands.append(int(exp))
            costs.append(float(cost))

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle("Algorithm Performance Comparison", fontsize=14)

        ax = axes[0][0]
        ax.bar(algos, times)
        ax.set_title("Runtime (ms)")

        ax = axes[0][1]
        ax.bar(algos, expands, color="orange")
        ax.set_title("Expanded Nodes")

        ax = axes[1][0]
        ax.bar(algos, costs, color="green")
        ax.set_title("Path Cost")

        ax = axes[1][1]
        ax.axis("off")
        summary = ""
        for a, t, e, c in zip(algos, times, expands, costs):
            summary += f"{a}: {t} ms | {e} expanded | cost={c}\n"
        ax.text(0.05, 0.5, summary, fontsize=10)

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # -------------------------------------------------------
    # DRAW GRID
    # -------------------------------------------------------
    def draw_grid(self, path=None):
        self.ax.clear()

        # İlk açılışta grid yoksa, sadece boş beyaz alan çiz
        if self.gridmap is None:
            self.ax.set_facecolor("white")
            self.ax.set_title("Grid Map (no data yet)")
            self.canvas.draw()
            return

        grid = self.gridmap.grid
        self.ax.imshow(grid, cmap="binary")

        # Start / Goal
        r0, c0 = self.gridmap.start
        r1, c1 = self.gridmap.goal
        self.ax.scatter(c0, r0, c="green", s=80)
        self.ax.scatter(c1, r1, c="red", s=80)

        # Path
        if path:
            xs = [c for (r, c) in path]
            ys = [r for (r, c) in path]
            self.ax.plot(xs, ys, c="blue", linewidth=2)

        self.ax.set_title("Grid Map")
        self.ax.invert_yaxis()
        self.canvas.draw()


# ---------------------------------------------------------
# RUN GUI
# ---------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = RoutePlannerGUI(root)
    root.mainloop()
