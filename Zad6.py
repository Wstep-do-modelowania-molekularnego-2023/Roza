import numpy as np
from scipy.optimize import fmin
import matplotlib.pyplot as plt

k = 1.28E-23
kcal_to_J = 4184
R = 8.31446
T = 298


def Z_fixed(x):
    return x**6 - 3 * 1.6**2 * x**4 + (9* 1.6**4 - 5) * x**2 / 4 + 10/64

x_list = np.linspace(-2.3, 2.3, 1000)
y_list = []
for x in x_list:
    y_list.append(Z_fixed(x))
y_min = min(y_list)
y_max = max(y_list) - y_min
y_list = [(i - y_min) / y_max for i in y_list]
plt.plot(x_list, y_list, 'r')
plt.xlabel("Odległość międzyatomowa [A]")
plt.ylabel("Potencjał [kcal/mol]")

# Stosunek obstadzeń z rozkładu Boltzmanna w temperaturze pokojowej

x_min1 = float(fmin(Z_fixed, -3))
x_min2 = float(fmin(Z_fixed, 0))
x_min3 = float(fmin(Z_fixed, 3))
n_12 = np.exp(-abs(Z_fixed(x_min1) - Z_fixed(x_min2)) * kcal_to_J / (R*T))
n_13 = np.exp(-abs(Z_fixed(x_min1) - Z_fixed(x_min3)) * kcal_to_J / (R*T))

print("Stusunek obsadzeń w temperaturze pokojowej z rozkładu Boltzmanna: 1.000000 : {:.6f} : {:.6f}".format(n_12, n_13))


## Monte - Carlo

def boltzmann(x):
    p = np.exp(- (Z_fixed(x) - x_min1) * kcal_to_J / (R*T))
    return p

x = 0
x_mc = []

while len(x_mc) < 10000:
    # x_probe = x + 0.25 * np.random.choice([-1, 1])
    x_probe = x + float(np.random.uniform(-2, 2, 1)) # losujemy wielkość kroku. Muszę losować z szerokiego zakresu, żeby układ był w stanie przeskoczyć minimum
    if boltzmann(x_probe) > boltzmann(x):
        x = x_probe
    else:
        k = float(np.random.uniform(0, 1, 1))
        if k < boltzmann(x_probe):
            x = x_probe
        else:
            pass
    x_mc.append(x)

def counter(list, min, max):
    ctr = 0
    for i in list:
        if min <= i <= max:
            ctr += 1
    return ctr

bars = np.linspace(-2.5, 2.5, 10)
x_freq = [counter(x_mc, i, i+1) for i in bars]
max_freq = max(x_freq)
x_freq = [i / max_freq for i in x_freq]
plt.bar(bars, x_freq)
plt.show()

central = counter(x_mc, -1.1, 1.1)
print("Stosunek populacji stanów między kolejnymi maksimami: {:.6f} : {:.6f} : {:.6f}".format(counter(x_mc, -10, -1.1) / central, 1, counter(x_mc, 1.1, 10) / central))
