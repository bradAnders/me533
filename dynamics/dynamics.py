import sympy as sp
from sympy.utilities.lambdify import lambdify
from typing import Optional, List


class AbstractDynamics:

    def __init__(self, states: [str], time_variant=False):

        self._logged_vars = []
        self._state_str = states
        self._state_sym = [sp.symbols(x) for x in states]
        self._num_states = len(self._state_sym)
        self._time_variant = time_variant
        self._t = sp.symbols('t') if time_variant else None

    @property
    def logged_vars(self):
        return self._logged_vars

    @property
    def num_states(self):
        return self._num_states

    @property
    def state_str(self):
        return self._state_str

    @property
    def state_sym(self):
        return self._state_sym

    @property
    def time_variant(self):
        return self._time_variant

    @property
    def t(self):
        return self._t


class AbstractDynamicController(AbstractDynamics):

    def __init__(self, outputs: [str], **kwargs):
        super(AbstractDynamicController, self).__init__(**kwargs)

        self._outputs_str = outputs
        self._outputs_sym = [sp.symbols(u) for u in outputs]
        self._num_outputs = len(self._outputs_sym)

    @property
    def num_outputs(self):
        return self._num_outputs

    @property
    def outputs_str(self):
        return self._outputs_str

    @property
    def outputs_sym(self):
        return self._outputs_sym

    def control_law(self, x, t=None) -> []:
        raise NotImplementedError

    def get_compiled_controller(self, dynamic_states):
        if self.time_variant:
            return lambdify([dynamic_states, self.t], self.control_law(x=dynamic_states, t=self.t))

        else:
            return lambdify(dynamic_states, self.control_law(x=dynamic_states))


class AbstractDynamicModel(AbstractDynamics):

    def __init__(self, inputs: [str], initial_conditions: Optional[List[float]] = None, **kwargs):
        super(AbstractDynamicModel, self).__init__(**kwargs)

        self._input_str = inputs
        self._input_sym = [sp.symbols(u, real=True) for u in inputs]
        self._num_inputs = len(self._input_sym)

        self._initial_conditions = [0] * self._num_states
        if initial_conditions:
            self._initial_conditions[0:len(initial_conditions)] = initial_conditions

    @property
    def num_inputs(self):
        return self._num_inputs

    @property
    def input_sym(self):
        return self._input_sym

    @property
    def input_str(self):
        return self._input_str

    @property
    def initial_conditions(self):
        return self._initial_conditions

    def equations_of_motion(self, states, controls, t: Optional[float] = None):
        raise NotImplementedError

    def get_compiled_eom(self):
        if self.time_variant:
            return lambdify([*self.state_sym, *self.input_sym, self.t], self.equations_of_motion(
                states=self.state_sym,
                controls=self.input_sym,
                t=self.t,
            ))

        else:
            return lambdify([*self.state_sym, *self.input_sym], self.equations_of_motion(
                states=self.state_sym,
                controls=self.input_sym,
            ))
