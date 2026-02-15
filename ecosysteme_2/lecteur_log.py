import pygame
import random
import math

GRID_SIZE = 25          # nombre de cases par côté
TILE_SIZE = 22          # pixels par case
SIDEBAR_WIDTH = 100
SIDEBAR_BG = (255, 255, 255)
TEXT_COLOR = (20, 20, 20)
BG_COLOR = (30, 30, 30)   

COLOR_FOURMILIERE = (200, 100, 50)     # couleur de la fourmilière
COLOR_FOOD = (50, 50,0)   # couleur de la nourriture
COLOR_FOURMI = (45, 160, 60)     # couleur des fourmis
COLOR_MUR = (100, 100, 100) # couleur des murs
COLOR_PHEROMONE = (255, 0, 255) # couleur des phéromones
GRID_LINE_COLOR = (0, 0, 0)
GRID_LINE_WIDTH = 1
OUTER_BORDER_COLOR = (0, 0, 0)
OUTER_BORDER_WIDTH = 2

gen_to_record = [0, 1, 30, 50, 99, 199, 299, 399, 499, 799, 999, 1999, 4999]



def generate_tilemap(): #loader la carte
    
    f = open("ecosysteme_2/carte_fourmiliere.txt", "r")
    tiles = []
    tiles = [i.split(" ") for i in f.readlines()]
    f.close()
    return tiles

def draw_grid(screen, tiles, pheromones):#dessiner le terrain
    h = len(tiles)
    w = len(tiles[0])
    for j in range(h):
        for i in range(w):
            if tiles[i][j] == '1':
                color = COLOR_FOURMILIERE
            elif tiles[i][j] == '-1':
                color =  COLOR_MUR
            else:
                #couleur en gradiant en fonction de la quantité de phéromones
                intensite = pheromones[i][j]
                color = (min(255, int(COLOR_PHEROMONE[0] * intensite)), min(255, int(COLOR_PHEROMONE[1] * intensite)), min(255, int(COLOR_PHEROMONE[2] * intensite)))


            try:
                rect = pygame.Rect(j * TILE_SIZE, i * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(screen, color, rect)
            except:
                print(color)
            
    outer = pygame.Rect(0, 0, w * TILE_SIZE, h * TILE_SIZE)
    pygame.draw.rect(screen, OUTER_BORDER_COLOR, outer, OUTER_BORDER_WIDTH)

def draw_food(screen, position, quantity): #afficher la nourriture

    y, x = position
    color = (min(255, int(COLOR_FOOD[0] * quantity)), min(255, int(COLOR_FOOD[1] * quantity)), min(255, int(COLOR_FOOD[2] * quantity)))
    pygame.draw.circle(screen, color, ((x+1/2)*TILE_SIZE, (y+1/2)*TILE_SIZE), TILE_SIZE/3)

def draw_agent(screen, position, charge): #afficher un agent
    y, x = position
    
    pygame.draw.circle(screen, COLOR_FOURMI, ((x+1/2)*TILE_SIZE, (y+1/2)*TILE_SIZE), TILE_SIZE/3)
    if charge > 0:
        pygame.draw.circle(screen, COLOR_FOOD, ((x+1/2)*TILE_SIZE, (y+1/2)*TILE_SIZE), TILE_SIZE/5)

def draw_sidebar(screen, left_width, height, lines, *, title="Infos"): #afficher la barre d'infos
    sidebar_rect = pygame.Rect(left_width, 0, SIDEBAR_WIDTH, height)
    pygame.draw.rect(screen, SIDEBAR_BG, sidebar_rect)
    pygame.font.init()
    line_font  = pygame.font.Font(None, 24)
    y = 50

    for line in lines:
        surf = line_font.render(line, True, TEXT_COLOR)
        screen.blit(surf, (left_width + 12, y))
        y += 22 #valeur au pif pour l'espacement entre les lignes


from read_log2 import read_log

log_pos_agents, log_pheromones, log_pos_food,  log_charges = read_log("logs/logbest.txt")
N_STEPS = len(log_pos_agents)


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
        draw_grid(screen, tiles, log_pheromones[T])

        for l in range(len(log_pos_food[T])):
            for c in range(len(log_pos_food[T][l])):
                if log_pos_food[T][l][c] > 0:
                    draw_food(screen, (l, c), log_pos_food[T][l][c])

        N_AGENTS = len(log_pos_agents[T])
        for id in range(N_AGENTS):
            draw_agent(screen, log_pos_agents[T][id], log_charges[T][id])
            x,y = log_pos_agents[T][id]
            if tiles[y][x] == -1 :
                print("alerte")
        T+=1
        if T >= N_STEPS:
            T = 0

        grid_px = GRID_SIZE * TILE_SIZE
        info_lines = [
        f"Step: {T}"]
        draw_sidebar(screen, grid_px, grid_px, info_lines, title="Plateau")
        pygame.display.flip()
        clock.tick(3)
    pygame.quit()

if __name__ == "__main__":
    main()
