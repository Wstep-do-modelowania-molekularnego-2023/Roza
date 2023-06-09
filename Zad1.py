import matplotlib.pyplot as plt
import init_cond_and_algorithms as algs
import time


# Running the given algorithm
xn, pn = 1, 0
# x_plot, y_plot = 0, 0
x_list = []
p_list = []
t_list = []

Fn = algs.F(xn)

if algs.method == "verlet":
    for i in range (0, algs.n_steps):
        x_list.append(xn)
        p_list.append(pn)
        t_list.append(i * algs.delta_t)
        xn1, pn1, Fn1 = algs.verlet(xn, pn, Fn, algs.delta_t)
        xn = xn1
        pn = pn1
        Fn = Fn1

elif algs.method == "leapfrog":
    pn_minus_pol = pn - Fn * algs.delta_t / 2
    for i in range (0, algs.n_steps):
        x_list.append(xn)
        p_list.append(pn_minus_pol)
        t_list.append(i * algs.delta_t)
        xn1, pn_pol = algs.leap_frog(xn, pn_minus_pol, algs.delta_t)
        xn = xn1
        pn_minus_pol = pn_pol

elif algs.method == "euler":
    for i in range(0, algs.n_steps):
        x_list.append(xn)
        p_list.append(pn)
        t_list.append(i * algs.delta_t)
        xn1, pn1 = algs.euler(xn, pn, algs.delta_t)
        xn = xn1
        pn = pn1

print("Execution time: ", time.time() - algs.start_time, "s")


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
plt.savefig("x_on_t")
plotter(t, p)
plt.savefig("y_on_t")
plotter(x, p)
plt.savefig("phase_space")
