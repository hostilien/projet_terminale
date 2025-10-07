# perlin_grid.py
# Affiche une grille carrée (Pygame) basée sur un bruit de Perlin 2D.
# Cases vertes si le bruit dépasse un seuil, sinon blanches.
# Aucune lib externe au-delà de pygame nécessaire.

import pygame
import random
import math

# ---------- Paramètres ----------
GRID_SIZE = 32          # nombre de cases par côté (grille GRID_SIZE x GRID_SIZE)
TILE_SIZE = 22          # pixels par case
SCALE = 0.08            # échelle du bruit (plus petit => motifs plus grands)
OCTAVES = 4             # nombre d'octaves (superpositions de bruit)
PERSISTENCE = 0.5       # influence décroissante des octaves
LACUNARITY = 2       # fréquence croissante des octaves
SEED = random.randint(0, 10000000)              # pour reproduire la même carte
THRESHOLD = 0.08        # seuil Perlin pour le vert (ajuste pour +/− de vert)
SHOW_GRID_LINES = False # True pour dessiner les contours


SIDEBAR_WIDTH = 100
SIDEBAR_BG = (255, 255, 255)
TEXT_COLOR = (20, 20, 20)


BG_COLOR = (30, 30, 30)         # fond de fenêtre
COLOR_GREEN = (45, 160, 60)     # terrain "herbe"
COLOR_WHITE = (230, 230, 230)   # terrain "clair"
SHOW_GRID_LINES = True          # ← active le contour de chaque case
GRID_LINE_COLOR = (0, 0, 0)
GRID_LINE_WIDTH = 1

OUTER_BORDER = True             # ← cadre autour de la carte
OUTER_BORDER_COLOR = (0, 0, 0)
OUTER_BORDER_WIDTH = 2
# ---------- Implémentation Perlin 2D (pur Python) ----------
# Référence: Perlin classique avec permutation, gradients 2D, fade curve.

class Perlin2D:
    def __init__(self, seed=None):
        rnd = random.Random(seed)
        p = list(range(256))
        rnd.shuffle(p)
        self.perm = p + p  # répétition pour éviter le wrap
        # gradients unitaires 2D (8 directions)
        self.grads = [
            ( 1, 0), (-1, 0), (0, 1), (0,-1),
            ( 1, 1), (-1, 1), (1,-1), (-1,-1)
        ]
        # normaliser diagonales
        self.grads = [(gx / math.hypot(gx, gy), gy / math.hypot(gx, gy)) for gx, gy in self.grads]

    @staticmethod
    def fade(t):
        # 6t^5 - 15t^4 + 10t^3 : interpolation douce de Perlin
        return t * t * t * (t * (t * 6 - 15) + 10)

    @staticmethod
    def lerp(a, b, t):
        return a + t * (b - a)

    def grad(self, hash_, x, y):
        g = self.grads[hash_ & 7]  # 8 gradients
        return g[0] * x + g[1] * y

    def noise(self, x, y):
        # coordonnées de la cellule
        xi = int(math.floor(x)) & 255
        yi = int(math.floor(y)) & 255
        # distances dans la cellule
        xf = x - math.floor(x)
        yf = y - math.floor(y)

        u = self.fade(xf)
        v = self.fade(yf)

        # indices permutation
        aa = self.perm[self.perm[xi] + yi]
        ab = self.perm[self.perm[xi] + yi + 1]
        ba = self.perm[self.perm[xi + 1] + yi]
        bb = self.perm[self.perm[xi + 1] + yi + 1]

        # contributions coins
        x1 = self.grad(aa, xf,     yf)
        x2 = self.grad(ba, xf - 1, yf)
        y1 = self.lerp(x1, x2, u)

        x1 = self.grad(ab, xf,     yf - 1)
        x2 = self.grad(bb, xf - 1, yf - 1)
        y2 = self.lerp(x1, x2, u)

        # interpolation finale
        n = self.lerp(y1, y2, v)
        # normaliser approximativement dans [-1,1]
        return max(-1.0, min(1.0, n))

def perlin_fbm(px, py, scale=0.05, octaves=4, persistence=0.5, lacunarity=2.0, perlin=None):
    """ Fractional Brownian Motion : somme d'octaves de Perlin. """
    if perlin is None:
        perlin = Perlin2D(SEED)
    amp = 1.0
    freq = 1.0
    total = 0.0
    max_amp = 0.0
    for _ in range(octaves):
        total += amp * perlin.noise(px * scale * freq, py * scale * freq)
        max_amp += amp
        amp *= persistence
        freq *= lacunarity
    return total / max_amp  # ~[-1,1]

# ---------- Carte de tuiles ----------
def generate_tilemap(n, scale, octaves, persistence, lacunarity, threshold, seed):
    """""
    per = Perlin2D(seed)
    tiles = [[1 for _ in range(n)] for _ in range(n)]  # 1 = vert, 0 = blanc
    for j in range(n):
        for i in range(n):
            # Option: centrer la carte pour éviter bords répétitifs
            x = i
            y = j
            val = perlin_fbm(x, y, scale, octaves, persistence, lacunarity, per)
            tiles[j][i] = 1 if val >= threshold else 0
    
    f = open("carte.txt", "w")
    for row in tiles:
        f.write(" ".join(str(cell) for cell in row) + "\n")
    f.close()
    return tiles
    """""
    f = open("carte.txt", "r")
    tiles = []
    tiles = [i.split(" ") for i in f.readlines()]
    f.close()
    return tiles

# ---------- Affichage Pygame ----------
def draw_grid(screen, tiles):
    h = len(tiles)
    w = len(tiles[0])
    for j in range(h):
        for i in range(w):
            if tiles[i][j] == '1':
                color = COLOR_GREEN
            else:
                color =  COLOR_WHITE

            rect = pygame.Rect(j * TILE_SIZE, i * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(screen, color, rect)
            if SHOW_GRID_LINES:
                pygame.draw.rect(screen, GRID_LINE_COLOR, rect, GRID_LINE_WIDTH)

    # Cadre extérieur autour de toute la grille
    if OUTER_BORDER:
        outer = pygame.Rect(0, 0, w * TILE_SIZE, h * TILE_SIZE)
        pygame.draw.rect(screen, OUTER_BORDER_COLOR, outer, OUTER_BORDER_WIDTH)
from pathlib import Path


def draw_food(screen, position):
    y, x = position
    pygame.draw.circle(screen, (250, 50, 50), ((x+1/2)*TILE_SIZE, (y+1/2)*TILE_SIZE), TILE_SIZE/3)

def draw_agent(screen, position, color):
    y, x = position
    pygame.draw.circle(screen, color, ((x+1/2)*TILE_SIZE, (y+1/2)*TILE_SIZE), TILE_SIZE/3)

def draw_sidebar(screen, left_width, height, lines, *, title="Infos"):
    # fond blanc
    sidebar_rect = pygame.Rect(left_width, 0, SIDEBAR_WIDTH, height)
    pygame.draw.rect(screen, SIDEBAR_BG, sidebar_rect)

    # texte
    if not pygame.font.get_init():
        pygame.font.init()
    line_font  = pygame.font.Font(None, 24)


    y = 50
    max_w = SIDEBAR_WIDTH - 24
    for line in lines:
        surf = line_font.render(line, True, TEXT_COLOR)
        screen.blit(surf, (left_width + 12, y))
        y += 22
        





def gen_game_random():
    pos_food = [(random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)) for i in range(30)]
    colors = [(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)) for i in range(10)]
    positions = [[(random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)) for i in range(10)]]
    moves = [[] for i in range(10)]
    for n_step in range(30):
        positions.append([])
        for id in range(10):
            move = [(0,0), (1, 0), (-1, 0), (0, 1), (0, -1)][random.randint(0, 4)]
            new_pos = (max(0, min(GRID_SIZE-1, positions[-2][id][0] + move[0])), max(0, min(GRID_SIZE-1, positions[-2][id][1] + move[1])))
            positions[-1].append(new_pos)
    return positions, colors, pos_food


from read_log import read_log
log_pos_agents, log_energies, log_pos_food = read_log("log499.txt")
colors = [(0,0,0)]+[(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)) for i in range(len(log_pos_agents[0])-1)]

def main():
    pygame.init()
    size = GRID_SIZE * TILE_SIZE
    screen = pygame.display.set_mode((size + SIDEBAR_WIDTH, size), pygame.RESIZABLE)

    tiles = generate_tilemap(
        GRID_SIZE, SCALE, OCTAVES, PERSISTENCE, LACUNARITY, THRESHOLD, SEED
    )

    clock = pygame.time.Clock()
    running = True
    T = 0
    while running:
        # --- Events (ici tu ajouteras les contrôles et l'animation plus tard) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
           

        # --- Draw ---

        screen.fill(BG_COLOR)
        draw_grid(screen, tiles)

        for f in log_pos_food[T]:
            draw_food(screen, f)
        N_AGENTS = len(log_pos_agents[T])
        for id in range(N_AGENTS):
            if log_energies[T][id]>0:
                draw_agent(screen, log_pos_agents[T][id], colors[id])
        T+=1
        if T >= len(log_pos_agents):
            T = 0
        # Exemple : afficher une image (coccinelle)
        """""
        for id in range(10):
            x, y = positions[T][id]
            #corriger l'affichage de l'image pour qu'elle soit centrée dans la case
            screen.blit(sprite, ((x-1/2)*TILE_SIZE-sprite.get_width()/4, (y-1/2)*TILE_SIZE-sprite.get_height()/4))
        
        """""
        grid_px = GRID_SIZE * TILE_SIZE
        info_lines = [
        f"Step: {T}"]
        draw_sidebar(screen, grid_px, grid_px, info_lines, title="Plateau")
        pygame.display.flip()
        clock.tick(2)



    pygame.quit()

if __name__ == "__main__":
    main()
