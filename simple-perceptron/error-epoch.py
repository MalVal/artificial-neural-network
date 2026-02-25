import sys
import numpy as np
import matplotlib.pyplot as plt

error_file = sys.argv[1]
errors = np.loadtxt(error_file, delimiter=",")

if errors.ndim == 1:
    epochs = [errors[0]]
    nb_errors = [errors[1]]
else:
    epochs = errors[:,0]
    nb_errors = errors[:,1]

plt.plot(epochs, nb_errors, marker='o')
plt.title("Nombre d'erreurs par epoque")
plt.xlabel("Epoque")
plt.ylabel("Nombre d'erreurs")
plt.grid(True)
plt.show()