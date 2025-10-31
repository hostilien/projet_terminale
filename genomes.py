carte = open("carte.txt", "r")
tiles = [i.split(" ") for i in carte.readlines()]
import numpy as np
import pickle
import os
import neat
import random
import matplotlib.pyplot as plt
N_RUNS = 25
N_STEPS = 400
L =32

def init_simulation(genomes):
    N_POPULATION = len(genomes)
    indices_positions = random.sample(range(L**2), N_POPULATION)
    positions = [(i//L, i%L) for i in indices_positions]
    map = [[0 for _ in range(L)] for _ in range(L)] # 0 = case vide, >0 = id agent
    tiles_fertiles = [(i, j) for i in range(L) for j in range(L) if tiles[i][j] == '1']
    food = random.sample(tiles_fertiles, 50)
    energies = [20.0 for _ in range(N_POPULATION)]


    for i in range(1, N_POPULATION+1):
        map[positions[i-1][0]][positions[i-1][1]] = i # 0 = case vide, >0 = id agent
    for (x, y) in food:
        map[x][y] = -1 # nourriture

    
    return positions, map, energies, food

def vision(pos):
    x, y = pos
    vision = []
    for i in range(x-1, x+2):
        for j in range(y-1, y+2):
            if 0 <= i < L and 0 <= j < L:
                if map[i][j] == 0:
                    vision.append(float(tiles[i][j]))
                elif map[i][j] == -1:
                    vision.append(2) # nourriture
                else:
                    vision.append(-1.0) # autre agent
            else:
                vision.append(-2.0) # mur
            # 0 vide, 1 fertile, 2 nourriture, -1 agent, -2 mur
    return vision



gen_to_record = [0, 1, 30, 50, 99, 199, 299, 399, 499, 799, 999, 1999, 4999]
G = 0


def simulation(genomes, config):
    
    global map, positions, G
    nets = [neat.nn.FeedForwardNetwork.create(genome, config) for _, genome in genomes]
    for genome_id, (_, genome) in enumerate(genomes):
        genome.fitness = 0.0
    life_time = [0 for _ in range(len(genomes))]
    if G in gen_to_record:
        log_score = [N_STEPS for _ in range(len(genomes))]
    for i in range(N_RUNS):
        positions, map, energies, food = init_simulation(genomes)
        dead = [False for _ in range(len(genomes))]
        register = (G in gen_to_record and i==0)
        if register:
            log_positions = []
            log_energies = []
            log_food = []
            log_positions.append(positions.copy())
            log_energies.append(energies.copy())
            log_food.append(food.copy())

        
        for step in range(N_STEPS):
            tiles_fertiles = [(i, j) for i in range(L) for j in range(L) if tiles[i][j] == '1']
            food = random.sample(tiles_fertiles, 10)

            for (x, y) in food:
                if map[x][y] == 0:
                    map[x][y] = -1

            for genome_id, (_, genome) in enumerate(genomes):
                if energies[genome_id] > 0:
                    amange = False

                    net = nets[genome_id]
                    input = vision(positions[genome_id])+[float(energies[genome_id])/30.0, step/N_STEPS, positions[genome_id][0]/float(L), positions[genome_id][1]/float(L)]
                    input = np.array(input)/2.0

                    output = net.activate(input)

                    move = output.index(max(output))
                    if move == 0: # up
                        new_position = (max(0, positions[genome_id][0]-1), positions[genome_id][1])
                    elif move == 1: # down
                        new_position = (min(L-1, positions[genome_id][0]+1), positions[genome_id][1])
                    elif move == 2: # left
                        new_position = (positions[genome_id][0], max(0, positions[genome_id][1]-1))
                    elif move == 3: # right
                        new_position = (positions[genome_id][0], min(L-1, positions[genome_id][1]+1))
                    else: # stay
                        new_position = positions[genome_id]
                    
                    if map[new_position[0]][new_position[1]] <= 0  : #on fait le mvt si la case ets vide
                        if new_position!= positions[genome_id]: #cout énérgétique si on bouge
                            energies[genome_id] -= 1.0
                        else:
                            energies[genome_id] -= 0.5 
                        if map[new_position[0]][new_position[1]] == -1: # il y avait de la nourriture
                            energies[genome_id] += 10.0
                    
                        map[positions[genome_id][0]][positions[genome_id][1]] = 0
                        positions[genome_id] = new_position
                        map[new_position[0]][new_position[1]] = genome_id+1 #plus 1 car 0 = case vide
                    else:
                        energies[genome_id] -= 0.5
                    
                    genome.fitness += 1.0
                    life_time[genome_id] += 1
                elif not dead[genome_id]:
                    map[positions[genome_id][0]][positions[genome_id][1]] = 0 # l'agent meurt
                    positions[genome_id] = (1000, 1000) # on l'envoie au cimetière, hors carte
                    if register:
                        log_score[genome_id] = step #faire gaffe si déjà mort pas augmenter score
                    dead[genome_id] = True
            if register:     
                log_positions.append(positions.copy())
                log_energies.append(energies.copy())
                food = []
                for x in range(L):
                    for y in range(L):
                        if map[x][y] == -1:
                            food.append((x, y))
                
                log_food.append(food.copy())

    for genome_id, (_, genome) in enumerate(genomes):
        genome.fitness /= N_RUNS
        life_time[genome_id] /= N_RUNS
    
    if G in gen_to_record:
        
        s = 0
        l_fitness = []
        for genome_id, (_, genome) in enumerate(genomes):
            s += life_time[genome_id]
            l_fitness.append(life_time[genome_id])

        plt.hist(l_fitness, bins=[i for i in range(0, N_STEPS, 10)])
        plt.title(f"Répartition des fitness - Génération {G}")
        plt.xlabel("Fitness")
        plt.ylabel("Nombre d'agents")
        plt.savefig(f"logs/fitness_gen_{G}.png")
        plt.clf()

        f = open("logs/log"+str(G)+".txt", "w")
        f.write("Generation "+str(G)+"\n")
        
        for t in range(len(log_positions)):
            for pos in log_positions[t]:
                f.write(f"{pos} ")
            f.write("\n")
            for e in log_energies[t]:
                f.write(f"{e} ")
            f.write("\n")

            for (x, y) in log_food[t]:
                f.write(f"{(x, y)} ")
            f.write("\n\n")

        for s in log_score:
            f.write(f"{s} ")

        f.close()
    G+=1




def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    checkpoint = neat.Checkpointer(
    generation_interval=10,
    filename_prefix='checkpoints/neat-checkpoint-')
    p.add_reporter(checkpoint)

   
    try:
        winner = p.run(simulation, 500)
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
    plt.savefig("logs/fitness_over_time.png")

    

from pathlib import Path
#config_path = Path(r"/Users/nilsdesurmont/Desktop/Informatique/projet_terminale/config_genomes.txt")
config_path = Path(r"C:\Users\cite scolaire 78\Documents\projet_terminale\config_genomes.txt")
config_path = str(config_path)

if __name__ == '__main__':

    local_dir = os.path.dirname(__file__)
    run(config_path)