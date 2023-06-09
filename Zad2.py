import matplotlib.pyplot as plt
import init_cond_and_algorithms as algs
import math


# Defining time steps basing on oscillation period
xn, pn = 1, 0
T = 2 * math.pi * math.sqrt(algs.M / algs.K)
sim_time = T * 10
Fn = algs.F(xn)

if algs.method == "verlet":
    for step in (500, 100, 50):     # Plotting E(t) for different value of oscillation period
        delta_t = T/step
        N_steps = int(sim_time/delta_t)     # Defining the number of steps, constant for all period steps
        x_list = []
        p_list = []
        t_list = []
        E_list = []
        E_mean = 0
        for i in range(0, N_steps):
            x_list.append(xn)
            p_list.append(pn)
            t_list.append(i * delta_t)
            E = pn ** 2 / (2 * algs.M) + algs.k * xn ** 2 / 2
            E_list.append(E)
            E_mean = E_mean * i / (i+1) + E * 1 / (i+1)
            xn1, pn1, Fn1 = algs.verlet(xn, pn, Fn, delta_t)
            xn = xn1
            pn = pn1
            Fn = Fn1
        plt.plot(t_list, E_list, label = f"T/{step}")
        plt.xlabel("Time, t")
        plt.ylabel("Total energy, E")
        print("Mean energy for delta_t = ", f"T/{step}: ", E_mean)
    plt.legend()
    plt.show()

    for step in (500, 100, 50, 25, 10, 5):  # Plotting p(x) for different value of oscillation period
        delta_t = T/step
        N_steps = int(sim_time/delta_t)     # Defining the number of steps, constant for all period steps
        x_list = []
        p_list = []
        t_list = []
        for i in range(0, N_steps):
            x_list.append(xn)
            p_list.append(pn)
            t_list.append(i * delta_t)
            xn1, pn1, Fn1 = algs.verlet(xn, pn, Fn, delta_t)
            xn, pn, Fn = xn1, pn1, Fn1
        plt.plot(x_list, p_list, label = f"T/{step}", linewidth = 0.3)
        plt.xlabel("Position, x")
        plt.ylabel("Momentum, p")
    plt.legend()
    plt.show()


elif algs.method == "euler":
    for step in (500, 200, 100):            # Plotting E(t) for different value of oscillation period
        delta_t = T/step
        N_steps = int(sim_time/delta_t)     # Defining the number of steps, constant for all period steps
        x_list = []
        p_list = []
        t_list = []
        E_list = []
        E_mean = 0
        for i in range(0, N_steps):
            x_list.append(xn)
            p_list.append(pn)
            t_list.append(i * delta_t)
            E = pn ** 2 / (2 * algs.M) + algs.K * xn ** 2 / 2
            E_list.append(E)
            E_mean = E_mean * i / (i+1) + E * 1 / (i+1)
            xn1, pn1 = algs.euler(xn, pn, delta_t)
            xn = xn1
            pn = pn1
        plt.plot(t_list, E_list, label = f"T/{step}")
        plt.xlabel("Time, t")
        plt.ylabel("Total energy, E")
        print("Mean energy for delta_T = {:.4f}: {:.4f}".format(1/step, E_mean))
    plt.legend()
    plt.show()

    for step in (500, 200, 100):             # Plotting p(x) for different value of oscillation period
        delta_t = T/step
        N_steps = int(sim_time/delta_t)     # Defining the number of steps, constant for all period steps
        x_list = []
        p_list = []
        t_list = []
        for i in range(0, N_steps):
            x_list.append(xn)
            p_list.append(pn)
            t_list.append(i * delta_t)
            xn1, pn1 = algs.euler(xn, pn, delta_t)
            xn = xn1
            pn = pn1
        plt.plot(x_list, p_list, label = f"T/{step}", linewidth = 0.6)
        plt.xlabel("Position, x")
        plt.ylabel("Momentum, p")
    plt.legend()
    plt.show()
