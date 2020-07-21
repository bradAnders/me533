import time
import pandas
import sympy as sp

from simulator import Simulator
from dynamics.dynamics import AbstractDynamicController
from dynamics.bicycle_model import BicycleModel
from dynamics.bicycle_model_linearized import BicycleModelLinearized


def sin_wave(amplitude=1.0, frequency=1.0, offset=0.0):
    t = sp.symbols('t')

    return amplitude * sp.sin(frequency * t) + offset


class OpenLoopController(AbstractDynamicController):

    def __init__(self, **kwargs):
        super(OpenLoopController, self).__init__(
            outputs=[
                'a_f',
                'a_r',
                'd_f',
                'd_r'],
            **kwargs
        )

    def control_law(self, x, t=None) -> []:
        w = 4.0
        return [
            0,
            20000.0 * (1 - sp.cos(t)) * sp.exp(-t),
            40.0 * sp.sin(t*w/2.0) * (1 - sp.cos(t*w)) / 2.0 * sp.pi / 180.0,
            0,
        ]


if __name__ == '__main__':

    program_start = time.time()

    # stiffness_N_deg = 2.25
    stiffness_N_deg = 2.25 * sp.pi / 180.0
    parameters = {
        'm': 230.0,  # kg
        'c_f': 2.25,  # stiffness_N_deg * sp.pi / 180.0,  # N / rad
        'c_r': 2.25,  # stiffness_N_deg * sp.pi / 180.0,  # N / rad
        'l_f': 0.69,  # m
        'l_r': 0.85,  # m
        'I_z': 1.0,  # N-m / rad/s
    }

    simulations = [
        ('bicycle_model_linearized', BicycleModelLinearized(parameters), OpenLoopController(time_variant=True)),
        ('bicycle_model_nonlinear', BicycleModel(parameters), OpenLoopController(time_variant=True)),
    ]

    for title, dynamics, controller in simulations:

        sim = Simulator(
            dynamics=dynamics,
            control_law=controller
        )

        sim_start = time.time()
        data, labels = sim.run(0.0, 10.0, 1000)
        sim_stop = time.time()

        print('    ' + title + ' simulation completed in %0.2f seconds' % (sim_stop - sim_start))

        df = pandas.DataFrame(
            data=data,
            columns=labels)

        df.to_csv(title + '.csv')

    program_stop = time.time()
    print('Program completed in %0.2f seconds' % (program_stop - program_start))
