from __future__ import division
from sympy import symbols, solve


N_parent = 10**12
N_no_migration = 10**8
N_migration = 10**9

x_parent = 0.3
x_migration = 0.01
x_no_migration = 0.001

D_transfer = 0.005
D_parent = 0.01

# assumes left hand side is equal to zero
K = symbols('K')
function = ((K/(x_migration*N_migration))-1)/((K/(x_no_migration*N_no_migration))-1) - ((K/(D_transfer*x_migration*N_migration + D_parent*x_parent*N_parent))-1)/((K/(D_transfer*x_no_migration*N_no_migration))-1)
#function = ((K/(D_transfer*x_migration*N_migration + D_parent*x_parent*N_parent))-1)/((K/(D_transfer*x_no_migration*N_no_migration))-1)

sol = solve(function)[0]
print(function)
print(sol)
