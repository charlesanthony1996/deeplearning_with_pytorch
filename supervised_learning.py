import numpy as np
import matplotlib.pyplot as plt

x = np.array([0.03, 0.19, 0.34, 0.46, 0.78, 0.81, 1.08, 1.18, 1.39, 1.60, 1.65, 1.90])
y = np.array([0.67, 0.85, 1.05, 1.0, 1.40, 1.5, 1.3, 1.54, 1.55, 1.68, 1.73, 1.6 ])

print(x)
print(y)

def f(x, phi0, phi1):
    y = x

    return y


def plot(x, y, phi0, phi1):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    plt.xlim([0, 2.0])
    plt.ylim([0, 2.0])
    ax.set_xlabel('Input x')
    ax.set_ylabel('Output y')

    x_line = np.arange(0, 2, 0.01)
    y_line = f(x_line, phi0, phi1)
    plt.plot(x_line, y_line, 'b--', lw = 2)

    plt.show()



