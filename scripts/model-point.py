import sys
import numpy as np
import matplotlib.pyplot as plt

# =============================
# Lecture des fichiers
# =============================
data_file = sys.argv[1]
model_file = sys.argv[2]

data = np.loadtxt(data_file, delimiter=",")
w0, w1, w2 = np.loadtxt(model_file, delimiter=",")

x1 = data[:,0]
x2 = data[:,1]
labels = data[:,2]

# =============================
# Calcul de la droite de séparation
# =============================
if w2 != 0:
    # Étendue des points + marge
    x_min, x_max = x1.min() - 0.5, x1.max() + 0.5
    x_vals = np.linspace(x_min, x_max, 200)
    y_vals = (-w0 - w1*x_vals) / w2
    plt.plot(x_vals, y_vals, color='green', label="Frontière de décision")
else:
    # Cas où w2 = 0 -> droite verticale
    x_vert = -w0 / w1
    plt.axvline(x=x_vert, color='green', label="Frontière de décision (verticale)")

# =============================
# Affichage des points
# =============================
plt.scatter(x1, x2, c=labels, cmap="bwr", edgecolors="k", s=80)
plt.title("Perceptron - Frontière de décision")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(True)
plt.show()