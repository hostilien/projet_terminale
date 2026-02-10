import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ast


# ---------- Parsing ----------
def parse_positions(line):
    tokens = line.strip().split(" ")
    positions = []
    for t in tokens:
        if t:
            positions.append(ast.literal_eval(t))  # "(x, y)" -> tuple
    return positions


def parse_float_line(line):
    return np.array([float(x) for x in line.strip().split(" ") if x])


def load_log(path):
    with open(path, "r") as f:
        lines = f.readlines()

    positions_steps = []
    pheromones_steps = []
    food_steps = []

    i = 0
    while i < len(lines):
        if lines[i].strip() == "":
            i += 1
            continue

        pos = parse_positions(lines[i])
        phe = parse_float_line(lines[i + 1])
        food = parse_float_line(lines[i + 2])

        positions_steps.append(pos)
        pheromones_steps.append(phe)
        food_steps.append(food)

        i += 4  # positions, pheromones, food, blank line

    return positions_steps, pheromones_steps, food_steps


# ---------- GUI ----------
class LogViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Log Viewer")
        self.geometry("1000x700")

        self.positions = None
        self.pheromones = None
        self.food = None

        top = tk.Frame(self)
        top.pack(fill=tk.X)

        tk.Button(top, text="Ouvrir log", command=self.open_file).pack(side=tk.LEFT, padx=5)

        self.step = tk.IntVar(value=0)
        self.slider = tk.Scale(
            top, from_=0, to=0, orient=tk.HORIZONTAL,
            variable=self.step, command=lambda _: self.redraw()
        )
        self.slider.pack(fill=tk.X, expand=True, padx=10)

        self.fig, self.ax = plt.subplots(figsize=(7, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def open_file(self):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if not path:
            return

        self.positions, self.pheromones, self.food = load_log(path)
        self.slider.config(to=len(self.positions) - 1)
        self.step.set(0)
        self.redraw()

    def redraw(self):
        if self.positions is None:
            return

        i = self.step.get()
        self.ax.clear()

        # ----- pheromones -----
        phe = self.pheromones[i]
        size = int(np.sqrt(len(phe)))
        phe_map = phe.reshape(size, size)

        im = self.ax.imshow(phe_map, origin="lower")
        self.fig.colorbar(im, ax=self.ax, fraction=0.046)

        # ----- positions -----
        pos = self.positions[i]
        if pos:
            xs, ys = zip(*pos)
            self.ax.scatter(xs, ys, c="red", s=10, label="agents")

        # ----- food -----
        food = self.food[i]
        ys, xs = np.where(food.reshape(size, size) > 0)
        self.ax.scatter(xs, ys, c="green", s=30, marker="s", label="food")

        self.ax.set_title(f"Step {i}")
        self.ax.legend()
        self.canvas.draw()


if __name__ == "__main__":
    LogViewer().mainloop()
