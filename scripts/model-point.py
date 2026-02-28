import sys
import numpy as np
import matplotlib.pyplot as plt

data_file = sys.argv[1]
model_file = sys.argv[2]

data = np.loadtxt(data_file, delimiter=",")
w0, w1, w2 = np.loadtxt(model_file, delimiter=",")

x1 = data[:,0]
x2 = data[:,1]
labels = data[:,2]

# Droite de séparation
x_vals = np.linspace(-1, 2, 100)
y_vals = (-w0 - w1*x_vals)/w2
plt.plot(x_vals, y_vals)

# Affichage des points
plt.scatter(x1, x2, c=labels, cmap="bwr", edgecolors="k")

plt.title("Perceptron - Frontiere de decision")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()