import sys
import numpy as np
import matplotlib.pyplot as plt

data_file = sys.argv[1]
model_file = sys.argv[2]

data = np.loadtxt(data_file, delimiter=",")
weights = np.loadtxt(model_file, delimiter=",")

# Si un seul modèle, on le transforme en matrice 1xN
if weights.ndim == 1:
    weights = weights.reshape(1, -1)

x1 = data[:, 0]
x2 = data[:, 1]
labels = data[:, -1]

plt.scatter(x1, x2, c=labels, cmap="tab10", edgecolors="k", s=80)

x_vals = np.linspace(x1.min() - 0.5, x1.max() + 0.5, 200)

# tracer chaque frontière
for i, w in enumerate(weights):
    if len(w) != 3:
        continue

    w0, w1, w2 = w

    if w2 != 0:
        y_vals = (-w0 - w1 * x_vals) / w2
        plt.plot(x_vals, y_vals, label=f"Classe {i}")
    else:
        plt.axvline(x=-w0 / w1, label=f"Classe {i}")

plt.title("Frontières de décision (Perceptron monocouche)")
plt.xlabel("x1")
plt.ylabel("x2")

plt.legend()
plt.grid(True)
plt.show()