import tkinter as tk
from tkinter import filedialog, messagebox

# --- Configuration des tuiles ---
TILES = [
    ("Normale", 0, "#f2f2f2"),
    ("Fourmilière", 1, "#ffcc66"),
    ("Mur", 2, "#666666"),
    ("Fertile", 3, "#66cc66"),
]
CODE_TO_TILE = {code: (name, code, color) for (name, code, color) in TILES}

CELL_SIZE = 28  # taille d'une case en pixels


class MapEditor(tk.Tk):
    def __init__(self, rows=15, cols=25):
        super().__init__()
        self.title("Éditeur de map (export TXT)")

        self.rows = rows
        self.cols = cols

        self.current_tile_idx = 0  # index dans TILES
        self.grid_data = [[0 for _ in range(cols)] for _ in range(rows)]  # codes numériques

        # --- UI ---
        self._build_toolbar()
        self._build_canvas()
        self._redraw_all()

    def _build_toolbar(self):
        bar = tk.Frame(self, padx=8, pady=8)
        bar.pack(fill="x")

        self.tile_label = tk.Label(bar, text=self._tile_text(), width=28, anchor="w")
        self.tile_label.pack(side="left")

        tk.Button(bar, text="Tuile suivante (Espace)", command=self.next_tile).pack(side="left", padx=6)
        tk.Button(bar, text="Exporter TXT (Ctrl+S)", command=self.export_txt).pack(side="left", padx=6)
        tk.Button(bar, text="Charger TXT", command=self.load_txt).pack(side="left", padx=6)
        tk.Button(bar, text="Effacer (tout à 0)", command=self.clear_grid).pack(side="left", padx=6)

        size_frame = tk.Frame(bar)
        size_frame.pack(side="right")

        tk.Label(size_frame, text="Lignes").grid(row=0, column=0, padx=4)
        self.rows_var = tk.IntVar(value=self.rows)
        tk.Entry(size_frame, textvariable=self.rows_var, width=5).grid(row=0, column=1, padx=4)

        tk.Label(size_frame, text="Colonnes").grid(row=0, column=2, padx=4)
        self.cols_var = tk.IntVar(value=self.cols)
        tk.Entry(size_frame, textvariable=self.cols_var, width=5).grid(row=0, column=3, padx=4)

        tk.Button(size_frame, text="Redimensionner", command=self.resize_grid).grid(row=0, column=4, padx=6)

        # Raccourcis
        self.bind("<space>", lambda e: self.next_tile())
        self.bind("<Control-s>", lambda e: self.export_txt())

    def _build_canvas(self):
        self.canvas = tk.Canvas(
            self,
            width=self.cols * CELL_SIZE,
            height=self.rows * CELL_SIZE,
            bg="white",
            highlightthickness=0,
        )
        self.canvas.pack(padx=8, pady=8)

        self.canvas.bind("<Button-1>", self.on_left_click)   # place la tuile sélectionnée
        self.canvas.bind("<Button-3>", self.on_right_click)  # remet à normale (0)

    def _tile_text(self):
        name, code, _ = TILES[self.current_tile_idx]
        return f"Tuile sélectionnée : {name} (code {code}) | Clic gauche=placer, clic droit=0"

    def next_tile(self):
        self.current_tile_idx = (self.current_tile_idx + 1) % len(TILES)
        self.tile_label.config(text=self._tile_text())

    def clear_grid(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self.grid_data[r][c] = 0
        self._redraw_all()

    def resize_grid(self):
        new_r = int(self.rows_var.get())
        new_c = int(self.cols_var.get())
        if new_r <= 0 or new_c <= 0:
            messagebox.showerror("Erreur", "Lignes/colonnes doivent être > 0.")
            return

        new_grid = [[0 for _ in range(new_c)] for _ in range(new_r)]
        # copie de l'ancien contenu dans la nouvelle grille
        for r in range(min(self.rows, new_r)):
            for c in range(min(self.cols, new_c)):
                new_grid[r][c] = self.grid_data[r][c]

        self.rows, self.cols = new_r, new_c
        self.grid_data = new_grid

        self.canvas.config(width=self.cols * CELL_SIZE, height=self.rows * CELL_SIZE)
        self._redraw_all()

    def _cell_from_xy(self, x, y):
        c = x // CELL_SIZE
        r = y // CELL_SIZE
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return r, c
        return None

    def on_left_click(self, event):
        rc = self._cell_from_xy(event.x, event.y)
        if not rc:
            return
        r, c = rc
        _, code, _ = TILES[self.current_tile_idx]

        # Optionnel : n'autoriser qu'une seule fourmilière
        if code == 1:
            # efface les anciennes fourmilières
            for rr in range(self.rows):
                for cc in range(self.cols):
                    if self.grid_data[rr][cc] == 1:
                        self.grid_data[rr][cc] = 0

        self.grid_data[r][c] = code
        self._redraw_cell(r, c)

    def on_right_click(self, event):
        rc = self._cell_from_xy(event.x, event.y)
        if not rc:
            return
        r, c = rc
        self.grid_data[r][c] = 0
        self._redraw_cell(r, c)

    def _redraw_all(self):
        self.canvas.delete("all")
        for r in range(self.rows):
            for c in range(self.cols):
                self._draw_rect(r, c)

    def _redraw_cell(self, r, c):
        # redessine juste une case : on supprime l'ancien item taggé
        tag = f"cell_{r}_{c}"
        self.canvas.delete(tag)
        self._draw_rect(r, c, tag=tag)

    def _draw_rect(self, r, c, tag=None):
        code = self.grid_data[r][c]
        if code not in CODE_TO_TILE:
            code = 0
            self.grid_data[r][c] = 0

        _, _, color = CODE_TO_TILE[code]
        x1 = c * CELL_SIZE
        y1 = r * CELL_SIZE
        x2 = x1 + CELL_SIZE
        y2 = y1 + CELL_SIZE

        tags = (tag,) if tag else (f"cell_{r}_{c}",)
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="#cccccc", tags=tags)

        # petit texte du code (facultatif, retire si tu veux une grille plus propre)
        self.canvas.create_text(
            x1 + CELL_SIZE / 2,
            y1 + CELL_SIZE / 2,
            text=str(self.grid_data[r][c]),
            fill="#000000",
            font=("Arial", 10),
            tags=tags
        )

    def export_txt(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Fichier texte", "*.txt"), ("Tous les fichiers", "*.*")]
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                for r in range(self.rows):
                    line = " ".join(str(self.grid_data[r][c]) for c in range(self.cols))
                    f.write(line + "\n")
            messagebox.showinfo("Export", f"Map exportée dans :\n{path}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'exporter :\n{e}")

    def load_txt(self):
        path = filedialog.askopenfilename(
            filetypes=[("Fichier texte", "*.txt"), ("Tous les fichiers", "*.*")]
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]

            new_grid = []
            for ln in lines:
                parts = ln.split()
                row = [int(x) for x in parts]
                new_grid.append(row)

            # vérif rectangle
            cols = len(new_grid[0])
            if any(len(row) != cols for row in new_grid):
                raise ValueError("Le fichier n'est pas une grille rectangulaire.")

            self.rows, self.cols = len(new_grid), cols
            self.rows_var.set(self.rows)
            self.cols_var.set(self.cols)
            self.grid_data = new_grid

            self.canvas.config(width=self.cols * CELL_SIZE, height=self.rows * CELL_SIZE)
            self._redraw_all()
            messagebox.showinfo("Chargement", f"Map chargée depuis :\n{path}")

        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger :\n{e}")


if __name__ == "__main__":
    # Tu peux changer les dimensions par défaut ici
    app = MapEditor(rows=15, cols=25)
    app.mainloop()
