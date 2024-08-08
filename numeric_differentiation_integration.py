# 1. plot to show the difference between the analytical derivative and the numeric implementation
import numpy as np
import matplotlib.pyplot as plt
def f(x):
    '''Function equivalent to sin(2x), should work for one argument or a numpy array'''
    return np.sin(2*x)

def df_analytic(x):
    '''
    The analytic derivative
    '''
    return 2*np.cos(2*x)

def forward_difference(f, x, dx):
    '''
    This function implements the forward difference method for the
    first derivative of the function f at position x using interval
    dx.
    '''
    return (f(x+dx)-f(x))/dx

xs = np.linspace(-2*np.pi,2*np.pi,100)

df_dx_1 = forward_difference(f, xs, dx=1e-6)
df_dx_2 = forward_difference(f, xs, dx=1e-8)
df_dx_3 = forward_difference(f, xs, dx=1e-10)
df_dx_analytical = df_analytic(xs)

plt.figure(figsize=(10, 6))
plt.plot(xs, df_dx_1 - df_dx_analytical, label='too large')
plt.plot(xs, df_dx_2 - df_dx_analytical, label='about right')
plt.plot(xs, df_dx_3 - df_dx_analytical, label='too small')
plt.legend()
plt.xlabel("x values")
plt.ylabel("Analytical derivative - forward difference")
plt.title('Difference between the analytical derivative and the numerical implementation')
plt.show()

# 2. numerical integration using Simpson's rule -
# integrand is approximated by a quadratic function, which is then exactly integrated
# over the interval of interest (usually a small interval)

def f(x):
    '''Function equivalent to x^2 cos(2x).'''
    return (x**2)*np.cos(2*x)

def g(x):
    '''Analytical integral of f(x).'''
    return (1/4)*((((2*x**2)-1)*np.sin(2*x))+2*x*np.cos(2*x))

def integrate_analytic(xmin, xmax):
    '''Analytical integral of f(x) from xmin to xmax.'''
    return g(xmax) - g(xmin)

# derivation

import sympy
x, h, f1, f2, f3 = sympy.symbols("x h f1 f2 f3")
a, b, c = sympy.symbols("a b c")

def integrate_numeric(xmin, xmax, N):
    '''
    Numerical integral of f from xmin to xmax using Simpson's rule with
        N panels (even number required).
    '''
    dx = (xmax - xmin) / N # determine width of panel
    x_n = np.linspace(xmin, xmax, 2*N + 1)
    # need 2N+1 points since each interval defined by 2 endpoints, plus additional point in the middle to form quadratic polynomial

    sum = 0
    sum = sum + f(x_n[0])

    for i in range(1, len(x_n) - 1): # loop from 2nd element to 2nd to last

        if i % 2 == 0:
            sum = sum + 2 * f(x_n[i]) # even elements multiplied by 2
        if i % 2 != 0:
            sum = sum + 4 * f(x_n[i]) # odd elements multiplied by 4

    sum = sum + f(x_n[-1])

    return (dx / 6) * sum

# 3. log-log plot to show fractional error between numerical and analytical result as number of panels is varied

x0, x1 = 0, 2  # bounds to integrate f(x) over
panel_counts = [4, 8, 16, 32, 64, 128, 256, 512, 1024]  # panel numbers to use
result_analytic = integrate_analytic(x0, x1)  # define reference value from analytical solution

fractional_errors = np.zeros(9)

for i in range(len(panel_counts)):
    numeric = integrate_numeric(x0, x1, panel_counts[i])
    fractional_error = abs((numeric - result_analytic)/result_analytic)
    fractional_errors[i] = fractional_error

plt.figure(figsize=(8, 4))

plt.loglog(panel_counts, fractional_errors, marker='o')

plt.ylabel('log(Fractional error)', fontsize=15)
plt.xlabel('log(Number of panels)', fontsize=15)
plt.title('Difference between numerically calculated integral and analytically derived result', fontsize=15)

plt.show()
