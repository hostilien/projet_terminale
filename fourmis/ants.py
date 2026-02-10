import sys, math, random
import numpy as np

from PySide6 import QtWidgets, QtCore
import pyqtgraph as pg


# ---------- Simulation (simple, mais suffisante pour l'UI) ----------
class AntSim:
    def __init__(self, w=220, h=160, n_ants=600):
        self.w, self.h = w, h
        self.n_ants = n_ants

        self.nest = (w // 2, h // 2)

        self.food = np.zeros((h, w), dtype=np.float32)
        self.pher = np.zeros((h, w), dtype=np.float32)  # un seul champ pour l'exemple
        self.obst = np.zeros((h, w), dtype=np.uint8)

        # zone de nourriture
        self.food[25:45, 160:200] = 5.0

        # obstacle exemple
        self.obst[80:82, 30:160] = 1

        # paramètres (modifiables via UI)
        self.evap = 0.02
        self.diff = 0.20
        self.deposit = 1.0
        self.step_size = 1.0
        self.turn = math.pi / 6
        self.noise = 0.15

        # fourmis : x,y,theta,has_food
        self.x = np.full(n_ants, self.nest[0], dtype=np.float32) + np.random.uniform(-2, 2, n_ants)
        self.y = np.full(n_ants, self.nest[1], dtype=np.float32) + np.random.uniform(-2, 2, n_ants)
        self.theta = np.random.uniform(0, 2 * math.pi, n_ants).astype(np.float32)
        self.has_food = np.zeros(n_ants, dtype=np.bool_)

    def reset(self):
        self.__init__(self.w, self.h, self.n_ants)

    def _in_bounds(self, ix, iy):
        return (0 <= ix < self.w) and (0 <= iy < self.h)

    def _is_free(self, ix, iy):
        return self._in_bounds(ix, iy) and self.obst[iy, ix] == 0

    def _diffuse(self, field):
        up = np.roll(field, -1, axis=0)
        down = np.roll(field, 1, axis=0)
        left = np.roll(field, 1, axis=1)
        right = np.roll(field, -1, axis=1)
        avg = (up + down + left + right) * 0.25
        field[:] = (1 - self.diff) * field + self.diff * avg

        # corrige les bords (évite wrap-around)
        field[0, :] = field[1, :]
        field[-1, :] = field[-2, :]
        field[:, 0] = field[:, 1]
        field[:, -1] = field[:, -2]

    def step(self, substeps=1):
        for _ in range(substeps):
            # update phéromones
            self._diffuse(self.pher)
            self.pher *= (1.0 - self.evap)

            # déplacement (boucle simple : ok pour ~1000-5000 fourmis)
            for i in range(self.n_ants):
                x, y, th = float(self.x[i]), float(self.y[i]), float(self.theta[i])

                def sense(angle_offset, dist=6):
                    sx = int(round(x + math.cos(th + angle_offset) * dist))
                    sy = int(round(y + math.sin(th + angle_offset) * dist))
                    if not self._in_bounds(sx, sy) or self.obst[sy, sx]:
                        return -1e9
                    return float(self.pher[sy, sx])

                L = sense(-self.turn)
                F = sense(0.0)
                R = sense(+self.turn)

                if random.random() < self.noise:
                    th += random.uniform(-self.turn, self.turn)
                else:
                    if L > F and L > R:
                        th -= self.turn
                    elif R > F and R > L:
                        th += self.turn

                nx = x + math.cos(th) * self.step_size
                ny = y + math.sin(th) * self.step_size
                ix, iy = int(nx), int(ny)

                if self._is_free(ix, iy):
                    x, y = nx, ny
                else:
                    th += math.pi * (0.6 + 0.8 * random.random())

                cx, cy = int(x), int(y)
                if self._in_bounds(cx, cy):
                    # prendre nourriture
                    if (not self.has_food[i]) and self.food[cy, cx] > 0.1:
                        self.has_food[i] = True
                        self.food[cy, cx] = max(0.0, self.food[cy, cx] - 1.0)

                    # déposer au nid
                    if self.has_food[i] and (abs(cx - self.nest[0]) <= 2 and abs(cy - self.nest[1]) <= 2):
                        self.has_food[i] = False

                    # dépôt de phéromone (ex : seulement quand on a de la nourriture)
                    if self.has_food[i]:
                        self.pher[cy, cx] += self.deposit

                self.x[i], self.y[i], self.theta[i] = x, y, th


# ---------- GUI ----------
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ant simulation (PySide6 + pyqtgraph)")

        self.sim = AntSim()
        self.running = True

        # ---- Vue pyqtgraph
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.view = pg.GraphicsLayoutWidget()
        self.plot = self.view.addPlot()
        self.plot.setAspectLocked(True)
        self.plot.hideAxis("left")
        self.plot.hideAxis("bottom")

        self.img_item = pg.ImageItem()
        self.plot.addItem(self.img_item)

        self.ants_item = pg.ScatterPlotItem(size=3)
        self.plot.addItem(self.ants_item)

        # ---- Controls
        self.btn_play = QtWidgets.QPushButton("Pause")
        self.btn_reset = QtWidgets.QPushButton("Reset")

        self.speed = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.speed.setRange(1, 20)
        self.speed.setValue(4)

        self.ev = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.ev.setRange(0, 100)
        self.ev.setValue(int(self.sim.evap * 1000))  # échelle pratique

        self.df = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.df.setRange(0, 100)
        self.df.setValue(int(self.sim.diff * 100))

        self.chk_pher = QtWidgets.QCheckBox("Afficher phéromones")
        self.chk_pher.setChecked(True)

        # Layout
        controls = QtWidgets.QFormLayout()
        controls.addRow(self.btn_play, self.btn_reset)
        controls.addRow("Vitesse (substeps/tick)", self.speed)
        controls.addRow("Evap (×0.001)", self.ev)
        controls.addRow("Diffusion (×0.01)", self.df)
        controls.addRow(self.chk_pher)

        right = QtWidgets.QWidget()
        right.setLayout(controls)

        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self.view, 1)
        layout.addWidget(right)

        # Signals
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_reset.clicked.connect(self.do_reset)
        self.ev.valueChanged.connect(self.on_params)
        self.df.valueChanged.connect(self.on_params)

        # Timer render
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)  # ~60 FPS

        self.on_params()

    def on_params(self):
        self.sim.evap = self.ev.value() / 1000.0
        self.sim.diff = self.df.value() / 100.0

    def toggle_play(self):
        self.running = not self.running
        self.btn_play.setText("Pause" if self.running else "Play")

    def do_reset(self):
        self.sim.reset()

    def tick(self):
        if self.running:
            self.sim.step(substeps=self.speed.value())

        # Affichage : phéromones
        if self.chk_pher.isChecked():
            img = np.clip(self.sim.pher / 30.0, 0, 1)
        else:
            img = np.zeros_like(self.sim.pher)

        # superpose nourriture et obstacles dans l'image (simple & lisible)
        img = img.copy()
        img[self.sim.obst == 1] = 0.25
        img = np.maximum(img, np.clip(self.sim.food / 5.0, 0, 1) * 0.7)

        self.img_item.setImage(img, autoLevels=False)

        # Affichage : fourmis
        spots = [{"pos": (float(self.sim.x[i]), float(self.sim.y[i]))} for i in range(self.sim.n_ants)]
        self.ants_item.setData(spots)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(1100, 700)
    w.show()
    sys.exit(app.exec())
