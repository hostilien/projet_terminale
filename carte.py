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



# ---------- Carte de tuiles ----------
def generate_tilemap():
    
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
log_pos_agents, log_energies, log_pos_food = read_log("logs/log299.txt")
colors = [(0,0,0)]+[(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)) for i in range(len(log_pos_agents[0])-1)]

def main():
    pygame.init()
    size = GRID_SIZE * TILE_SIZE
    screen = pygame.display.set_mode((size + SIDEBAR_WIDTH, size), pygame.RESIZABLE)

    tiles = generate_tilemap()

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
