import sympy as sp
from sympy.utilities.lambdify import lambdify


t = sp.symbols('t')

a = sp.sin(t)
b = sp.cos(t)

c = [a, b]

d = lambdify(t, a)

print(d)
