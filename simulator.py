import numpy as np
from dynamics.dynamics import AbstractDynamicModel, AbstractDynamicController


class Simulator:

    def __init__(self, dynamics: AbstractDynamicModel, control_law: AbstractDynamicController):

        self._eom = dynamics
        self._ctrl = control_law

        self._num_states = dynamics.num_states
        self._num_controls = dynamics.num_inputs

        self._t_start = 0.0
        self._t_end = 1.0
        self._num_pts = 100

        self._solver = self.rk4
        self.last_sim = None

    @property
    def last_sim_labels(self):
        return ['Time [s]', *self._eom.state_str, *self._ctrl.outputs_str, *self._eom.logged_vars]

    @property
    def last_sim_data(self):
        return self.last_sim

    def run(self, t_start=0.0, t_end=1.0, num_pts=100):

        self._t_start = t_start
        self._t_end = t_end
        self._num_pts = num_pts
        time_list = np.linspace(t_start, t_end, num_pts)
        time_list = time_list.reshape((num_pts, 1))

        time_step = time_list[1] - time_list[0]
        states = np.zeros((num_pts, self._num_states))
        states[0, :] = self._eom.initial_conditions
        controls = np.zeros((num_pts, self._num_controls))
        logged_signals = None

        dynamics = self._eom.get_compiled_eom()
        controller = self._ctrl.get_compiled_controller(self._eom.state_sym)

        for i, t_i in enumerate(time_list):
            if self._ctrl.time_variant:
                controls[i, :] = controller(states[i, :], t_i)
            else:
                controls[i, :] = controller(states[i, :])

            if i < len(time_list)-1:
                states[i + 1, :], logged_values = self._solver(
                    dx_dt=dynamics,
                    dt=time_step,
                    x=states[i, :],
                    u=controls[i, :],
                    t=t_i,
                    time_variant=self._eom.time_variant)
            else:
                _, logged_values = self._solver(
                    dx_dt=dynamics,
                    dt=time_step,
                    x=states[i, :],
                    u=controls[i, :],
                    t=t_i,
                    time_variant=self._eom.time_variant)

            if logged_signals is None:
                logged_signals = np.zeros((num_pts, len(logged_values)))
            logged_signals[i, :] = logged_values

        self.last_sim = np.concatenate((time_list, states, controls, logged_signals), axis=1)

        return self.last_sim, self.last_sim_labels

    @staticmethod
    def rk4(dx_dt, dt, x, u, t, time_variant=False):

        if time_variant:
            k_1, var = dx_dt(*x, *u, t)
            k_2, _ = dx_dt(x + k_1 * dt / 2.0, u, t + dt / 2.0)
            k_3, _ = dx_dt(x + k_2 * dt / 2.0, u, t + dt / 2.0)
            k_4, _ = dx_dt(x + k_3 * dt, u, t + dt)

        else:
            k_1, var = dx_dt(*x, *u)
            k_2, _ = dx_dt(*x + np.multiply(k_1, (dt / 2.0)), *u)
            k_3, _ = dx_dt(*x + np.multiply(k_2, (dt / 2.0)), *u)
            k_4, _ = dx_dt(*x + np.multiply(k_3, dt), *u)

        return x + np.multiply(dt / 6.0, (k_1 + np.multiply(2.0, k_2) + np.multiply(2.0, k_3) + k_4)), var
