import time
import random
import sympy as sp
import math
import numpy as np
import matplotlib.pyplot as plt
from simulator import Simulator


def a_1(t):
    return random.uniform(-1, 1)


def a_2(t):
    return random.uniform(0, 2)


def b(t):
    return random.uniform(1, 4)


def sin_wave(amplitude=1.0, frequency=1.0, offset=0.0):

    t = sp.symbols('t')

    return amplitude * sp.sin(frequency * t) + offset


def ctrl_p7_10(x, x_d_funcs, t):

    lam = 1.0/50.0 * 10
    b_hat = 2.0
    k = 10.0

    x_d = [x_d_func(t) for x_d_func in x_d_funcs]

    x_e = [x_d[i] - x_i for i, x_i in enumerate(x)]

    s = x_e[2] + 2*lam*x_e[1] + lam**2*x_e[0]

    return 1/b_hat * (x_d[3] - 2*lam*x_e[2] - lam**2*x_d[1] - k*np.sign(s))


def dyn_p7_10(x, t, u, x_d):

    x_1_dot = x[1]
    x_2_dot = x[2]
    x_3_dot = - a_1(t) * x[2] ** 2 - a_2(t) * x[1] ** 5 * math.sin(4 * x[0]) + b(t) * u(x, x_d, t)

    return np.array([x_1_dot, x_2_dot, x_3_dot])


if __name__ == '__main__':

    start = time.time()
    sim = Simulator(
        initial_conditions=[0.0, 0.0, 0.0],
        trajectory=sin_wave(amplitude=2.0),
        dynamics=dyn_p7_10,
        control_law=ctrl_p7_10
    )
    (sim_time, states) = sim.run(0.0, 0.5)
    (_, x_d) = sim.trajectory
    stop = time.time()

    print('Took %0.2f seconds' % (stop-start))

    plt.plot(sim_time, x_d[0])
    plt.plot(sim_time, states[0, :])
    plt.show()
