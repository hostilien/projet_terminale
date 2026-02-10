import numpy as np
import pickle
import os
import neat
import random
import matplotlib.pyplot as plt
carte = open("/Users/nilsdesurmont/Desktop/Informatique/projet_terminale/ecosysteme_2/carte_fourmiliere.txt", "r")
tiles = [i.split(" ") for i in carte.readlines()]

N_RUNS = 1
N_STEPS = 100
L =25
N_FOOD_INIT = 5
SIZE_FOOD = 20
FACTEUR_EVAPORATION = 0.1
N_FOURMIS = 25
def add_food(map_food):
    x,y = random.randint(0,L-1), random.randint(0,L-1)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < L and 0 <= ny < L:
                if tiles[nx][ny] == '0':
                    map_food[nx][ny] += SIZE_FOOD # ajouter de la nourriture
    return map_food

def init_simulation():
    N_POPULATION = 25
    map1 = [[int(tiles[i][j]) for i in range(L)] for j in range(L)]
    map_food = [[0 for _ in range(L)] for _ in range(L)]
    map_pheromones = np.zeros((L, L))
    charge = [0 for _ in range(N_POPULATION)]
    indices_positions = [(i,j) for i in range(L) for j in range(L) if tiles[i][j] == '1'] #on commence sur la fourmilière
    for i in range(N_POPULATION):
        x, y = indices_positions[i]
        map1[x][y] = i+1  #les fourmis sont numérotées de 1 à N_POPULATION
    for _ in range(N_FOOD_INIT):
        map_food = add_food(map_food)

    return N_POPULATION, map1, map_food, map_pheromones, indices_positions, charge





def vision(pos):
    x, y = pos
    vision = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < L and 0 <= ny < L:
                if map1[nx][ny] > 0 or map1[nx][ny] == -1:
                    vision.append(-1)  # autre fourmi ou mur
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
    num_individu = map1[x][y]
    vision.append(charge[num_individu - 1])  # charge de la fourmi
    return vision



gen_to_record = [0, 1, 30, 50, 99, 199, 299, 399, 499, 799, 999, 1999, 4999]
G = 0


def simulation(genomes, config):
    global map1, map_food, map_pheromones, G, charge
    nets = [neat.nn.FeedForwardNetwork.create(genome, config) for _, genome in genomes]
    if G in gen_to_record:
        print("Recording generation ", G)

        log_charge = []
        log_positions = []
        log_pheromones = []
        log_food = []

    for genome_id, (_, genome) in enumerate(genomes):
        genome.fitness = 0.0

        for i_fourmiliere in range(N_RUNS):
            N_FOURMIS, map1, map_food, map_pheromones, positions, charge = init_simulation()

            

            for i_step in range(N_STEPS):

                if G in gen_to_record and i_fourmiliere==0:
                    log_charge.append(charge.copy())
                    log_positions.append(positions.copy())
                    log_pheromones.append(map_pheromones.flatten().copy())
                    log_food.append(np.array(map_food.copy()).flatten())

                # Évaporation des phéromones
                map_pheromones *= (1 - FACTEUR_EVAPORATION)
                for pos in positions:
                    x, y = pos
                    num_individu = map1[x][y]
                    if num_individu > 0: #espaces codées par 0, fourmis par 1 à N_FOURMIS
                        input_vision = vision((x, y))
                        output = nets[genome_id].activate(input_vision)
                        direction = output.index(max(output))
                        deplacement = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # haut, droite, bas, gauche
                        if direction==4:
                            dx, dy = (0, 0)  # ne rien faire
                        elif direction==5:
                            dx, dy = deplacement[random.randint(0,3)]  # déplacement aléatoire

                        else:
                            dx, dy = deplacement[direction]

                        nx, ny = x + dx, y + dy
                        if 0 <= nx < L and 0 <= ny < L and map1[nx][ny] == 0:
                            map1[x][y] = 0
                            map1[nx][ny] = num_individu
                            positions[num_individu - 1] = (nx, ny)
                            # Interaction avec la nourriture
                            if map_food[nx][ny] > 0 and charge[num_individu - 1] == 0:
                                charge[num_individu - 1] = 1
                                map_food[nx][ny] -= 1
                            elif tiles[nx][ny] == "1" and charge[num_individu - 1] == 1:
                                genome.fitness += 10.0
                                charge[num_individu - 1] = 0
                            # Dépôt de phéromones                                           
                        if charge[num_individu - 1] == 1:
                            map_pheromones[x][y] += 1.0
                        else:
        
                            map_pheromones[x][y] += 0.5
        genome.fitness /= N_RUNS
    if G in gen_to_record:

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
    G += 1

        
                
                
           
    
                    
    



def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)

    
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
   

    try:
        winner = p.run(simulation, 50)
    except neat.CompleteExtinctionException:
        print("extinctionc occured")
        winner = stats.best_genome()

    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)
    
    
    best_fitness_per_gen = stats.get_fitness_stat(max)
    mean_fitness_per_gen = stats.get_fitness_mean()
    plt.plot(best_fitness_per_gen, label="Best Fitness")
    plt.plot(mean_fitness_per_gen, label="Mean Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.savefig("fitness_over_time.png")

    

from pathlib import Path
#config_path = Path(r"/Users/nilsdesurmont/Desktop/Informatique/projet_terminale/config_genomes.txt")
config_path = Path(r"/Users/nilsdesurmont/Desktop/Informatique/projet_terminale/ecosysteme_2/config_genomes2.txt")
config_path = str(config_path)

if __name__ == '__main__':

    local_dir = os.path.dirname(__file__)
    run(config_path)