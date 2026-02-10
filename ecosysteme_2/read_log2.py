def traiter(ligne):
    ligne = ligne.split(" ")
    return ligne

def separer(ligne):    

    ligne = ligne.replace("(", "")
    ligne = ligne.replace(")", "")
    ligne = ligne.replace(",", "")
    ligne = ligne.split(" ")
    l =  [int(i) for i in ligne ]
    return [(l[i], l[i+1]) for i in range(0, len(l), 2)]

def read_log(filename):
    log = open(filename, "r")
    lines = log.readlines()
    del lines[-1]
    log_pos_fourmis = []
    log_pos_food = []
    log_pos_pheromones = []
    log_pos_charge = []
    for id_l in range(0, len(lines), 5):
        log_pos_fourmis.append(separer(lines[id_l][:-2]))
        log_pos_charge.append([int(i) for i in lines[id_l+3][:-2].split(" ")])

        food = (traiter(lines[id_l+2][:-2]))
        tableau_food = []
        for i in range(25):
            ligne_food = []
            for j in range(25):
                ligne_food.append(int(food[i*25+j]))
            tableau_food.append(ligne_food)
        log_pos_food.append(tableau_food)
        pheromone = (traiter(lines[id_l+1][:-2]))
        tableau_pheromones = []
        for i in range(25):
            ligne_pheromones = []
            for j in range(25):
                ligne_pheromones.append(float(pheromone[i*25+j]))
            tableau_pheromones.append(ligne_pheromones)
        log_pos_pheromones.append(tableau_pheromones)



    return log_pos_fourmis, log_pos_pheromones, log_pos_food, log_pos_charge


