def traiter(ligne):
    ligne = ligne.replace("(", "")
    ligne = ligne.replace(")", "")
    ligne = ligne.replace(",", "")
    ligne = ligne.split(" ")
    
    l =  [int(i) for i in ligne]
    
    return [(l[i], l[i+1]) for i in range(0, len(l), 2)]

def read_log(filename):
    log = open(filename, "r")
    lines = log.readlines()
    log_pos_agents = []
    log_pos_food = []
    log_energies = []
    for id_l in range(1, len(lines), 4):
        log_pos_agents.append(traiter(lines[id_l][:-2]))
        log_energies.append([float(i) for i in lines[id_l+1][:-2].split(" ")])
        log_pos_food.append(traiter(lines[id_l+2][:-2]))
    return log_pos_agents, log_energies, log_pos_food
