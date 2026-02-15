import neat
import pickle
import networkx as nx
import matplotlib.pyplot as plt

def draw_neat_network(genome, config, node_names=None):
 
    G = nx.DiGraph()

    for node_key in genome.nodes:
        G.add_node(node_key)

    for cg in genome.connections.values():
        if cg.enabled:
            src, dst = cg.key
            G.add_edge(src, dst, weight=cg.weight)

    input_nodes = list(config.genome_config.input_keys)
    output_nodes = list(config.genome_config.output_keys)
    hidden_nodes = list(set(genome.nodes.keys()) - set(input_nodes) - set(output_nodes))

    color_map = []
    for node in G.nodes():
        if node in input_nodes:
            color_map.append("lightgreen")
        elif node in output_nodes:
            color_map.append("lightblue")
        else:
            color_map.append("orange")

    def layer_positions(nodes, y, x_start=0):
        """Retourne les positions (x, y) des nœuds d'une même couche"""
        if len(nodes) == 0:
            return {}
        spacing = 2.0
        x_offset = -(len(nodes) - 1) * spacing / 2
        return {n: (x_start + i * spacing + x_offset, y) for i, n in enumerate(sorted(nodes))}

    pos = {}
    pos.update(layer_positions(input_nodes, y=0))
    pos.update(layer_positions(hidden_nodes, y=2))
    pos.update(layer_positions(output_nodes, y=4))

    if node_names is None:
        labels = {n: str(n) for n in G.nodes()}
    else:
        labels = {n: node_names.get(n, str(n)) for n in G.nodes()}

    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, labels=labels,
            node_color=color_map, node_size=1200,
            arrows=True, font_size=8)

    labels_edges = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos,
        edge_labels={k: f"{v:.2f}" for k, v in labels_edges.items()},
        font_size=6, label_pos=0.4)

    plt.title("Réseau NEAT (entrées → sorties)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()



with open("winner.pkl", "rb") as f:
    winner = pickle.load(f)

config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    "config_genomes2.txt"
)

node_names = {-1 : "hautGauche", -2:"haut", -3:"hautDroite", -4:"gauche", -5:"case", -6:"droite", -7:"basGauche", -8: "bas", -9: "basDroitePh", -10 : "hautGauchePh", -11:"hautPh", -12:"hautDroite", -13:"gauchePh", -14:"casePh", -15:"droitePh", -16:"basGauchePh", -17: "basPh", -18: "basDroitePh", -19:"x", -20:"y", -21: "charge", 0: "haut", 1:"droite", 2:"bas", 3:"gauche", 4: "attendre", 5:"Rnd"}

 


draw_neat_network(winner, config, node_names)
