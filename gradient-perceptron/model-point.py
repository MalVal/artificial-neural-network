import sys
import numpy as np
import matplotlib.pyplot as plt

# Récupérer les chemins depuis les arguments
data_file = sys.argv[1]
model_file = sys.argv[2]

# Chargement des points et modèle
data = np.loadtxt(data_file, delimiter=",")
w0, w1, w2 = np.loadtxt(model_file, delimiter=",")

x1 = data[:,0]
x2 = data[:,1]
labels = data[:,2]

class0 = data[labels == -1]
class1 = data[labels == 1]

plt.scatter(class0[:,0], class0[:,1])
plt.scatter(class1[:,0], class1[:,1])

x_vals = np.linspace(-1, 2, 100)
y_vals = (-w0 - w1*x_vals)/w2
plt.plot(x_vals, y_vals)

plt.title("Perceptron - Frontiere de decision")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()