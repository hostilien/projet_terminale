import random
import math
GRID_SIZE = 32          # nombre de cases par côté (grille GRID_SIZE x GRID_SIZE)
TILE_SIZE = 22          # pixels par case
SCALE = 0.08            # échelle du bruit (plus petit => motifs plus grands)
OCTAVES = 4             # nombre d'octaves (superpositions de bruit)
PERSISTENCE = 0.5       # influence décroissante des octaves
LACUNARITY = 2       # fréquence croissante des octaves
THRESHOLD = 0.08  
SEED  = input("Seed (entier) ? ")


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
    
    per = Perlin2D(seed)
    tiles = [[1 for _ in range(n)] for _ in range(n)]  # 1 = vert, 0 = blanc
    for j in range(n):
        for i in range(n):
            # Option: centrer la carte pour éviter bords répétitifs
            x = i
            y = j
            val = perlin_fbm(x, y, scale, octaves, persistence, lacunarity, per)
            tiles[j][i] = 1 if val >= threshold else 0
    
    f = open("cartes/carte"+str(SEED)+".txt", "w")
    for row in tiles:
        f.write(" ".join(str(cell) for cell in row) + "\n")
    f.close()
    return tiles

generate_tilemap(GRID_SIZE, SCALE, OCTAVES, PERSISTENCE, LACUNARITY, THRESHOLD, SEED)

   