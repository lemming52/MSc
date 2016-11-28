import function
import numpy as np

def euler_integrate(function, timestep, count, initial):

    results = np.zeros(count)
    results[0] = initial

    for i in range(1, count):
        results[i] = results[i-1] + timestep * function.eval(results[i - 1])

    return results

def xy_euler_integrate(xfunc, yfunc, initial, timestep, count):

    results = np.zeros((2, count))
    results[0][0] = initial[0]
    results[1][0] = initial[1]

    for i in range(1, count):
        results[0][i] = results[0][i-1] + timestep * xfunc.eval([results[0][i - 1],
                                                                 results[1][i - 1]])
        results[1][i] = results[1][i-1] + timestep * yfunc.eval([results[0][i - 1],
                                                                 results[1][i - 1]])

    return results
