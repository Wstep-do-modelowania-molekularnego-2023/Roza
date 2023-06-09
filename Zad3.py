import matplotlib.pyplot as plt
import init_cond_and_algorithms as algs
import numpy as np


m, k = 1.139E-26, 1860
delta_t = 9E-17
xn, pn = 1, 0
x_list = []
p_list = []
t_list = []
equil = []

Fn = algs.F(xn, k)

if algs.method == "verlet":
    for i in range (0, algs.n_steps):
        x_list.append(xn)
        p_list.append(pn)
        t_list.append(i * delta_t)
        xn1, pn1, Fn1 = algs.verlet(xn, pn, Fn, delta_t, m, k)
        if i > 2 and np.sign(xn1) != np.sign(xn):
            equil.append(abs((t_list[-1] + t_list[-2]) / 2))
        xn = xn1
        pn = pn1
        Fn = Fn1
    period = sum([equil[i+2] - equil[i] for i in range(0, len(equil)- 2)]) / (len(equil)- 2)
    print("Frequency = ", 1 / period, "Hz")

elif algs.method == "leapfrog":
    pn_minus_pol = pn - Fn * delta_t / 2
    for i in range (0, algs.n_steps):
        x_list.append(xn)
        p_list.append(pn_minus_pol)
        t_list.append(i * delta_t)
        xn1, pn_pol = algs.leap_frog(xn, pn_minus_pol, delta_t, m, k)
        if i > 2 and np.sign(xn1) != np.sign(xn):
            equil.append(abs((t_list[-1] + t_list[-2]) / 2))
        xn = xn1
        pn_minus_pol = pn_pol
    period = sum([equil[i+2] - equil[i] for i in range(0, len(equil)- 2)]) / (len(equil)- 2)
    print("Frequency = ", 1 / period, "Hz")

elif algs.method == "euler":
    for i in range(0, algs.n_steps):
        x_list.append(xn)
        p_list.append(pn)
        t_list.append(i * delta_t)
        xn1, pn1 = algs.euler(xn, pn, delta_t, m, k)
        if i > 2 and np.sign(xn1) != np.sign(xn):
            equil.append(abs((t_list[-1] + t_list[-2]) / 2))
        xn = xn1
        pn = pn1
    period = sum([equil[i+2] - equil[i] for i in range(0, len(equil)- 2)]) / (len(equil)- 2)
    print("Frequency = {:.2e} Hz".format(1 / period))


# Plotting a selected dependence
x = dict(data = x_list, label = "Position, x")
p = dict(data = p_list, label = "Momentum, p")
t = dict(data = t_list, label = "Time, t")


def plotter(y, z):
    plt.plot(y["data"], z["data"])
    plt.xlabel(y["label"])
    plt.ylabel(z['label'])
    plt.show()
    plt.clf()


plotter(t, x)

