import numpy as np
import pickle
import os
import neat
import random
import matplotlib.pyplot as plt
import time
carte = open("ecosysteme_2/carte_fourmiliere.txt", "r")
carte = [i.split(" ") for i in carte.readlines()]

N_RUNS = 10
N_STEPS = 150
L =25
N_FOOD_INIT = 5
SIZE_FOOD = 3
FACTEUR_EVAPORATION = 0.02
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


gen_to_record = [0, 1, 30, 50, 99, 199, 299, 399, 499, 799, 999, 1999, 4999]
G = 0

def eval(genome, config):
    N_runs = 5
    fitness = 0
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    for i_run in range (N_runs):
            N_FOURMIS, tiles,map_agents, map_food, map_pheromones, positions, charge = init_simulation()
            for i_step in range(N_STEPS):
                map_pheromones *= (1 - FACTEUR_EVAPORATION)
                for pos in positions:
                    x, y = pos
                    num_individu = map_agents[x][y]
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
                    vision.append(charge[num_individu - 1])
                    
                    output = net.activate(vision)
                    choix_pheromone = output[6]
                    direction = output.index(max(output))
                    deplacement = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # haut, droite, bas, gauche
                    if direction==4:
                        dx, dy = (0, 0)  # ne rien faire
                    elif direction==5:
                        dx, dy = deplacement[random.randint(0,3)]  # déplacement aléatoire

                    elif direction!= 6:

                        dx, dy = deplacement[direction]

                    nx, ny = x + dx, y + dy
                    if 0 <= nx < L and 0 <= ny < L and map_agents[nx][ny] == 0 and tiles[nx][ny] != -1:  # vérifier que la case est libre et pas un mur
                        map_agents[x][y] = 0
                        map_agents[nx][ny] = num_individu
                        positions[num_individu - 1] = (nx, ny)
                        # Interaction avec la nourriture
                        if map_food[nx][ny] > 0 and charge[num_individu - 1] == 0:
                            charge[num_individu - 1] = 1
                            fitness += 2.0  # manger de la nourriture augmente la fitness
                            map_food[nx][ny] -= 1
                        elif tiles[nx][ny] == 1 and charge[num_individu - 1] == 1:
                            fitness += 15.0
                            charge[num_individu - 1] = 0
                        elif charge[num_individu - 1] == 1:
                            fitness-=0.1  # déposer des phéromones en portant de la nourriture
                        # Dépôt de phéromones   
                    map_pheromones[x][y] += choix_pheromone
                                    
                    """if charge[num_individu - 1] == 1:
                        map_pheromones[x][y] += 1.0
                    else:
    
                        map_pheromones[x][y] += 0.5"""
    fitness /= N_RUNS
    return fitness

import multiprocessing
import neat
import pickle
import matplotlib.pyplot as plt
import os
from pathlib import Path

def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # run les simulations en parallele
    n_workers = max(1, multiprocessing.cpu_count() - 1)
    pe = neat.ParallelEvaluator(n_workers, eval)

    try:
        winner = p.run(pe.evaluate, 100) 
    except neat.CompleteExtinctionException:
        print("extinction occured")
        winner = stats.best_genome()

    with open("ecosysteme_2/winner.pkl", "wb") as f:
        pickle.dump(winner, f)
        print("Winner enregsitré")

    best_fitness_per_gen = stats.get_fitness_stat(max)
    mean_fitness_per_gen = stats.get_fitness_mean()
    plt.plot(best_fitness_per_gen, label="Best Fitness")
    plt.plot(mean_fitness_per_gen, label="Mean Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.savefig("fitness_over_time.png")


config_path = r"ecosysteme_2/config_genomes2.txt"
config_path = str(config_path)

if __name__ == "__main__":
    run(config_path)