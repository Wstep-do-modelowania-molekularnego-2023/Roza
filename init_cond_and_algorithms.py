import sys
import time
start_time = time.time()

# Defining the default parameters
M, K = 1, 1
n_steps = 1000
delta_t = 0.05
method = "euler"

# Gathering information from markers
labels = ["-m", "-k", "-x0", "-p0", "-n_steps", "-delta_t", "-method"]
params = ["x", "p", "t"]
opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
for i in opts:
    val = sys.argv[sys.argv.index(i)+1]
    if i == "-m":
        m = float(val)
    elif i == "-k":
        k = float(val)
    elif i == "-x0":
        xn = float(val)
    elif i == "-p0":
        pn = float(val)
    elif i == "-n_steps":
        n_steps = float(val)
    elif i == "-delta_t":
        delta_t = float(val)
    elif i == "-method":
        if val == "verlet":
            method = "verlet"
        elif val == "leapfrog":
            method = "leapfrog"
        elif val == "euler":
            method = "euler"
        else:
            print("Method incorrect. Select \"verlet\", \"leapfrog\" or \"euler\"")
            quit()
    else:
        print("Marker incorrect. Try \"-m\" for mass, \"-k\" for oscillator constant, \"-x0\" for initial position, \"-p0\" for initial momentum, \"-n_steps\" for number of simulation steps, \"-delta_t\" for time step or \"-method\" for algorithm.")
        quit()


# Defining the algorithms
def F(x, k = K):
    return -k * x

def verlet(xn, pn, Fn, delta_t, m = M, k = K):
    xn1 = xn + pn * delta_t / m + Fn * delta_t**2 / (2*m)
    Fn1 = F(xn1, k)
    pn1 = pn + (Fn1 + Fn) * delta_t / 2
    return(xn1, pn1, Fn1)

def leap_frog(xn, pn_minus_pol, delta_t, m = M, k = K):
    Fn = F(xn, k)
    pn_pol = pn_minus_pol + Fn * delta_t
    xn1 = xn + 1/m * pn_pol * delta_t
    return(xn1, pn_pol)

def euler(xn, pn, delta_t, m = M, k = K):
    Fn = F(xn, k)
    xn1 = xn + pn * delta_t / m + Fn * delta_t**2 / (2*m)
    pn1 = pn + Fn * delta_t
    return(xn1, pn1)
