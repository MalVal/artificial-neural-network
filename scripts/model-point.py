import sys
import numpy as np
import matplotlib.pyplot as plt

data_file = sys.argv[1]
model_file = sys.argv[2]

data = np.loadtxt(data_file, delimiter=",")
weights = np.loadtxt(model_file, delimiter=",")

# =============================
# Régression 1D : 2 poids (w0, w1)
# =============================
if len(weights) == 2:
    w0, w1 = weights
    x = data[:, 0]
    y = data[:, 1]

    x_vals = np.linspace(x.min() - 0.5, x.max() + 0.5, 200)
    y_vals = w0 + w1 * x_vals

    plt.scatter(x, y, color='blue', edgecolors='k', s=80, label="Données")
    plt.plot(x_vals, y_vals, color='green', label="Droite de régression")
    plt.title("Régression")
    plt.xlabel("x")
    plt.ylabel("y")

# =============================
# Classification 2D : 3 poids (w0, w1, w2)
# =============================
elif len(weights) == 3:
    w0, w1, w2 = weights
    x1 = data[:, 0]
    x2 = data[:, 1]
    labels = data[:, 2]

    if w2 != 0:
        x_vals = np.linspace(x1.min() - 0.5, x1.max() + 0.5, 200)
        y_vals = (-w0 - w1 * x_vals) / w2
        plt.plot(x_vals, y_vals, color='green', label="Frontière de décision")
    else:
        plt.axvline(x=-w0 / w1, color='green', label="Frontière (verticale)")

    plt.scatter(x1, x2, c=labels, cmap="bwr", edgecolors="k", s=80)
    plt.title("Classification")
    plt.xlabel("x1")
    plt.ylabel("x2")

else:
    print(f"Nombre de poids non supporté : {len(weights)}")
    sys.exit(1)

plt.legend()
plt.grid(True)
plt.show()