class Eleve:
    def __init__(self, nom, age):
        self.nom = nom
        self.age = age
    def baise(self, autre):
        print(f"{self.nom} baise avec {autre.nom} c'est très sale")
    def anniversaire(self):
        self.age +=1
smantha = Eleve("Smantha", 16)
adlaurent = Eleve("Adlaurent", 23)
smantha.baise(adlaurent)
print('intéréssant', smantha.age)
