import sympy as sp
from dynamics.dynamics import AbstractDynamicModel
from typing import Optional


class BicycleModel(AbstractDynamicModel):

    def __init__(self, parameters):
        super(BicycleModel, self).__init__(
            states=[
                'x',
                'x_dot',
                'y',
                'y_dot',
                'psi',
                'psi_dot',
                'X',
                'Y',
                'S',
            ],
            inputs=[
                'a_f',
                'a_r',
                'd_f',
                'd_r'])

        self._logged_vars = [
            'beta_f',
            'beta_r',
            'beta',
            'alpha_f',
            'alpha_r',
            'v',
        ]

        self.m = parameters.get('m')
        self.c_f = parameters.get('c_f')
        self.c_r = parameters.get('c_r')
        self.l_f = parameters.get('l_f')
        self.l_r = parameters.get('l_r')
        self.I_z = parameters.get('I_z')

    def equations_of_motion(self, states, controls, t: Optional[float] = None):

        # All variables are in body frame (except, which psi isn't used)

        # Inputs
        f_long_f = controls[0]
        f_long_r = controls[1]
        delta_f = controls[2]
        delta_r = controls[3]

        # States
        # x = states[0]
        x_dot = states[1]
        # y = states[2]
        y_dot = states[3]
        psi = states[4]
        psi_dot = states[5]

        # Helpers
        v = sp.sqrt(x_dot**2 + y_dot**2)

        beta_f = sp.atan2(y_dot + self.l_f * psi_dot, x_dot + 0.00000001)
        beta_r = sp.atan2(y_dot - self.l_r * psi_dot, x_dot + 0.00000001)
        beta = sp.atan2(self.l_r * sp.tan(beta_f) + self.l_f * sp.tan(beta_r), self.l_f + self.l_r)

        alpha_f = delta_f - beta_f
        alpha_r = delta_r - beta_r

        f_lat_f = 2.0 * self.c_f * alpha_f
        f_lat_r = 2.0 * self.c_r * alpha_r

        f_long_x_f = f_long_f * sp.cos(delta_f)
        f_long_y_f = f_long_f * sp.sin(delta_f)

        f_lat_x_f = - f_lat_f * sp.sin(delta_f)
        f_lat_y_f = f_lat_f * sp.cos(delta_f)

        f_long_x_r = f_long_r * sp.cos(delta_r)
        f_long_y_r = f_long_r * sp.sin(delta_r)

        f_lat_x_r = - f_lat_r * sp.sin(delta_r)
        f_lat_y_r = + f_lat_r * sp.cos(delta_r)

        # Equations of Motion

        x_rate = (
            v * sp.cos(beta)
        )
        x_dot_rate = (
            (f_long_x_f + f_lat_x_f + f_long_x_r - f_lat_x_r) / self.m
            + psi_dot * y_dot
        )

        y_rate = (
            v * sp.sin(beta)
        )
        y_dot_rate = (
            (f_long_y_f + f_lat_y_f + f_long_y_r + f_lat_y_r) / self.m
            - psi_dot * x_dot
        )

        psi_rate = (
            v * sp.cos(beta) * (sp.tan(beta_f) - sp.tan(beta_r)) / (self.l_f + self.l_r)
        )
        psi_dot_rate = ((
            (f_long_y_f + f_lat_y_f) * self.l_f
            - (f_long_y_r + f_lat_y_r) * self.l_r
        ) / self.I_z)

        # Path construction

        s_rate = (
            v
        )
        x_fixed_rate = (
            v * sp.cos(psi + beta)
        )
        y_fixed_rate = (
            v * sp.sin(psi + beta)
        )

        return [
            x_rate,
            x_dot_rate,
            y_rate,
            y_dot_rate,
            psi_rate,
            psi_dot_rate,
            x_fixed_rate,
            y_fixed_rate,
            s_rate,
        ], [
            beta_f,
            beta_r,
            beta,
            alpha_f,
            alpha_r,
            v,
        ]
