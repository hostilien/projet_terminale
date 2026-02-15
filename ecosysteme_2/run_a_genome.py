import numpy as np
import pickle
import os
import neat
import random
import matplotlib.pyplot as plt
import time
carte = open("ecosysteme_2/carte_fourmiliere.txt", "r")
carte = [i.split(" ") for i in carte.readlines()]
G=9999
N_STEPS = 150
L =25
N_FOOD_INIT = 5
SIZE_FOOD = 3
FACTEUR_EVAPORATION = 0.1
N_FOURMIS = 4
def add_food(map_food):
    x,y = random.randint(0,L-1), random.randint(0,L-1)
    while (x**2+y**2)**(1/2) < 8: 
        x,y = random.randint(0,L-1), random.randint(0,L-1)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < L and 0 <= ny < L:
                if carte[nx][ny] == '0':
                    map_food[nx][ny] += SIZE_FOOD # ajouter de la nourriture
    return map_food

def init_simulation():
    N_POPULATION = 4
    tiles = [[int(carte[i][j]) for j in range(L)] for i in range(L)]
    map_agents = [[0 for _ in range(L)] for _ in range(L)] # 0 = case vide, >0 = id agent
    map_food = [[0 for _ in range(L)] for _ in range(L)]
    map_pheromones = np.zeros((L, L))
    charge = [0 for _ in range(N_POPULATION)]
    indices_positions = [(i,j) for i in range(L) for j in range(L) if tiles[i][j] == 1] #on commence sur la fourmilière
    for i in range(N_POPULATION):
        x, y = indices_positions[i]
        map_agents[x][y] = i+1  #les fourmis sont numérotées de 1 à N_POPULATION
    for _ in range(N_FOOD_INIT):
        x,y = random.randint(0,L-1), random.randint(0,L-1)
        while (x**2+y**2)**(1/2) < 8: 
            x,y = random.randint(0,L-1), random.randint(0,L-1)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < L and 0 <= ny < L:
                    if carte[nx][ny] == '0':
                        map_food[nx][ny] += SIZE_FOOD
    return N_POPULATION, tiles, map_agents, map_food, map_pheromones, indices_positions, charge

def vision(pos):
    x, y = pos
    vision = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < L and 0 <= ny < L:
                if tiles[nx][ny] != 0:
                    vision.append(tiles[nx][ny])  
                elif map_agents[nx][ny] > 0:
                    vision.append(-1)  # autre agent
                elif map_food[nx][ny] > 0:
                    vision.append(map_food[nx][ny])  # nourriture
                else:
                    vision.append(0) # case vide
            
            else:
                vision.append(-1)  # mur

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < L and 0 <= ny < L:
                vision.append(map_pheromones[nx][ny])
            else:
                vision.append(-1)  # mur
    vision.append(x)
    vision.append(y)
    num_individu = map_agents[x][y]
    vision.append(charge[num_individu - 1])  # charge de la fourmi
    return vision


def simulation(net):
    global tiles, map_agents, map_food, map_pheromones, G, charge
    log_charge = []
    log_positions = []
    log_pheromones = []
    log_food = []
    N_FOURMIS, tiles,map_agents, map_food, map_pheromones, positions, charge = init_simulation()
    for i_step in range(N_STEPS):
        log_charge.append(charge.copy())
        log_positions.append(positions.copy())
        log_pheromones.append(map_pheromones.flatten().copy())
        log_food.append(np.array(map_food.copy()).flatten())

        # Évaporation des phéromones
        map_pheromones *= (1 - FACTEUR_EVAPORATION)
        for pos in positions:
            x, y = pos
            num_individu = map_agents[x][y]
            if num_individu > 0: #espaces codées par 0, fourmis par 1 à N_FOURMIS
                input_vision = vision((x, y))
                
                output = net.activate(input_vision)
                direction = output.index(max(output))
                deplacement = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # haut, droite, bas, gauche
                if direction==4:
                    dx, dy = (0, 0)  # ne rien faire
                elif direction==5:
                    dx, dy = deplacement[random.randint(0,3)]  # déplacement aléatoire

                elif direction!=6:

                    dx, dy = deplacement[direction]

                nx, ny = x + dx, y + dy
                if 0 <= nx < L and 0 <= ny < L and map_agents[nx][ny] == 0 and tiles[nx][ny] != -1:  # vérifier que la case est libre et pas un mur
                    map_agents[x][y] = 0
                    map_agents[nx][ny] = num_individu
                    positions[num_individu - 1] = (nx, ny)
                    # Interaction avec la nourriture
                    if map_food[nx][ny] > 0 and charge[num_individu - 1] == 0:
                        charge[num_individu - 1] = 1
                        map_food[nx][ny] -= 1
                    elif tiles[nx][ny] == 1 and charge[num_individu - 1] == 1:
                        charge[num_individu - 1] = 0                                       

                map_pheromones[x][y] += output[6]
    log = open("logs/log"+str(G)+".txt", "w")
    for step in range(N_STEPS):
        for pos in log_positions[step]:
            log.write(str(pos)+" ")
        log.write("\n")

        for case in log_pheromones[step]:
            log.write(str(round(case, 3))+ " ")
        log.write("\n")

        for case in log_food[step]:
            log.write(str(case)+ " ")
        log.write("\n")

        for charge in log_charge[step]:
            log.write(str(charge)+ " ")
        log.write("\n")
        log.write("\n")
    log.close()
    print("simulation effectuée")

        

def run(config_file):
    #Importer le genome#
    WINNER_PATH = "ecosysteme_2/winner.pkl"
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )
    # Charger le genome
    with open(WINNER_PATH, "rb") as f:
        winner_genome = pickle.load(f)
    genome = neat.nn.FeedForwardNetwork.create(winner_genome, config)
    simulation(genome)

    
from pathlib import Path
config_path = r"ecosysteme_2/config_genomes2.txt"
config_path = str(config_path)


if __name__ == '__main__':

    local_dir = os.path.dirname(__file__)
    run(config_path)
