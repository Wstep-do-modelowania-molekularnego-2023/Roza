import numpy as np
from scipy.optimize import fmin
import matplotlib.pyplot as plt

k = 1.28E-23
kcal_to_J = 4184
R = 8.31446
T = 298

def Z(x, a, epsilon, y_Nz_prim, y_Nz):
    return a * x**4 + 6 * a * epsilon**2 * x**2 + y_Nz_prim * x + y_Nz

def Z_fixed(x):
    return 0.04 * (x**4 - 12*x**2 + 5*x + 0.2)


def plot_potential():
    x_list = np.linspace(-5, 4, 1000)
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

    x_min1 = float(fmin(Z_fixed, 0))
    x_min2 = float(fmin(Z_fixed, 3))
    n = np.exp(-abs(Z_fixed(x_min1) - Z_fixed(x_min2)) * kcal_to_J / (R*T))
    print("Stusunek obsadzeń w temperaturze pokojowej z rozkładu Boltzmanna: {:.6f}".format(n))
    return x_list, y_list


## Monte - Carlo

def boltzmann(x, x_next):
    p = np.exp(- (Z_fixed(x_next) - Z_fixed(x)) * kcal_to_J / (R*T))
    return min(p,1)

def gen_x_mc(num=10000):
    x = 0
    x_mc = []
    i = 0
    while i < num:
        # x_probe = x + 0.25 * np.random.choice([-1, 1])
        x_probe = x + float(np.random.normal(0, 0.5, 1)) # losujemy wielkość kroku. Muszę losować z szerokiego zakresu, żeby układ był w stanie przeskoczyć minimum
        p = boltzmann(x, x_probe)
        k = float(np.random.uniform(0, 1, 1))
        if k < p:
            x = x_probe
            x_mc.append(x)
            i += 1
    return x_mc

def gen_x_mc_2(num = 10000):
    x_mc = []
    x_min1 = float(fmin(Z_fixed, 0))
    for x in np.random.uniform(-5, 5, 10000):
        p = boltzmann(x_min1, x)
        k = float(np.random.uniform(0, 1, 1))
        if k < p:
            x_mc.append(x)
        else:
            pass
    return x_mc

def counter(list, min, max):
    ctr = 0
    for i in list:
        if min <= i <= max:
            ctr += 1
    return ctr

if __name__ == '__main__':
    plot_potential()
    x_mc = gen_x_mc()
    plt.hist(x_mc, 20, density = True)
    plt.show()
    print("Stosunek populacji stanów x > 0 do x < 0 z symulacji Monte-Carlo, metoda 1: ", counter(x_mc, 0, 10) / counter(x_mc, -10, 0))

    plot_potential()
    x_mc = gen_x_mc_2()
    plt.hist(x_mc, 20, density=True)
    plt.show()
    print("Stosunek populacji stanów x > 0 do x < 0 z symulacji Monte-Carlo, metoda 2: ", counter(x_mc, 0, 10) / counter(x_mc, -10, 0))
