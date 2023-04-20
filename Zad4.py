import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

n_points = 10000
n_fields = 100

def jedn_f(x, k):
    return k

def gauss_f(x, mu, sig):
    return np.exp(-np.power(x - mu, 2) / (2 * np.power(sig, 2))) / (sig * np.sqrt(2 * np.pi))

def count(list, min, max):
    counter = 0
    for i in list:
        if i > min and i < max:
            counter += 1
    return counter

jedn = np.random.uniform(0, 1, n_points)
gauss = np.random.normal(0.5, 0.3, size = n_points)
freqs_jedn = []
freqs_gauss = []

for i in range(0, n_fields):
    freqs_jedn.append(count(jedn, i/n_fields, (i+1)/n_fields))
    freqs_gauss.append(count(gauss, i/n_fields, (i+1)/n_fields))


for i in range(0, len(freqs_jedn)):
    freqs_jedn[i] = freqs_jedn[i] / n_fields

for i in range(0, len(freqs_gauss)):
    freqs_gauss[i] = freqs_gauss[i] / n_fields

x_list = np.linspace(0, 1, n_fields)

# Wykreślenie odpowiednuch krzywych i punktów wylosowanych z danych rozkładów

plt.plot(x_list, [jedn_f(i, 1) for i in x_list])
plt.plot(x_list, freqs_jedn, 'o')
plt.title("Rozkład jednostajny")
plt.ylim(0, 2)
plt.show()

plt.plot(x_list, [gauss_f(i, 0.5, 0.3) for i in x_list])
plt.plot(x_list, freqs_gauss, 'o')
plt.title("Rozkład normalny")
plt.show()

# Dopasowanie krzywych do wylosowanych danych i wyznaczenie błędów średniokwadratowych

par, cov = curve_fit(jedn_f, x_list, freqs_jedn)
print("Fitting parameters for uniform: ", par)
print("Mean squared error for uniform fitting: {:.6f}".format(np.mean((jedn_f(x_list, *par) - freqs_jedn)**2)))
# print("Mean squared error for uniform fitting: {:.6f}".format(np.mean([(jedn_f(x_list[i], 1) - freqs_jedn[i])**2 for i in range(0, len(x_list))])))

plt.plot(x_list, [jedn_f(i, *par) for i in range(0, len(x_list))])
plt.plot(x_list, freqs_jedn, 'o')
plt.title("Fitowanie prostej do punktów wylosowanych z rozkładu jednorodnego")
plt.ylim(0, 2)
plt.show()

par, cov = curve_fit(gauss_f, x_list, freqs_gauss)
print("Fitting parameters for gaussian: ", par)
print("Mean squared error for gaussian fitting: {:.6f}".format(np.mean((gauss_f(x_list, *par) - freqs_gauss)**2)))

plt.plot(x_list, gauss_f(x_list, *par))
plt.plot(x_list, freqs_gauss, 'o')
plt.title("Fitowanie gaussianu do punktów wylosowanych z rozkładu normalnego")
plt.show()
