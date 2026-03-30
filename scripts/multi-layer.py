import sys
import numpy as np
import matplotlib.pyplot as plt

# =========================
# ARGUMENTS
# =========================
grid_file = sys.argv[1]
train_file = sys.argv[2]

# =========================
# LOAD DATA
# =========================
grid = np.loadtxt(grid_file, delimiter=',')
train = np.loadtxt(train_file, delimiter=',')

# =========================
# DETECTION DIMENSIONS
# =========================
# grid format:
# 1D  -> x, y_pred
# 2D  -> x, y, o1, o2, ...

if grid.shape[1] == 2:
    # REGRESSION
    nb_inputs = 1
    nb_outputs = 1
else:
    # CLASSIFICATION 2D
    nb_inputs = 2
    nb_outputs = grid.shape[1] - 2

# =========================
# CAS REGRESSION (1D)
# =========================
if nb_inputs == 1:
    X = train[:, 0]
    Y = train[:, 1]

    x_grid = grid[:, 0]
    y_pred = grid[:, 1]

    # sort pour affichage propre
    idx = np.argsort(x_grid)
    x_grid = x_grid[idx]
    y_pred = y_pred[idx]

    plt.figure()
    plt.scatter(X, Y, label="Données")
    plt.plot(x_grid, y_pred, label="Prédiction MLP")
    plt.title("Régression")
    plt.legend()
    plt.grid()

# =========================
# CAS CLASSIFICATION (2D)
# =========================
elif nb_inputs == 2:

    X = train[:, :2]

    # labels réels
    if train.shape[1] > 3:
        Y = np.argmax(train[:, 2:], axis=1)
    else:
        Y = train[:, 2]

    x = grid[:, 0]
    y = grid[:, 1]
    outputs = grid[:, 2:]

    # prédictions (argmax)
    if outputs.shape[1] > 1:
        Z = np.argmax(outputs, axis=1)
    else:
        Z = (outputs[:, 0] > 0.5).astype(int)

    # reshape grille
    xi = np.unique(x)
    yi = np.unique(y)

    Z = Z.reshape(len(xi), len(yi))

    plt.figure()

    # zones de décision
    plt.contourf(xi, yi, Z.T, alpha=0.3)

    # points d'entraînement
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k')

    plt.title("Zones de décision")
    plt.grid()

# =========================
# AUTRE CAS
# =========================
else:
    print("Visualisation non supportée pour >2 dimensions")
    sys.exit(1)

# =========================
# AFFICHAGE
# =========================
plt.show()