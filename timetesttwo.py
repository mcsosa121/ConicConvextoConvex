from GeneralCaseTolerance import *
import matplotlib.pyplot as plt
import timeit
import math
import numpy
import cvxpy

# isclose is used to check if two floating point numbers are close
# enough to be considered equal.
def isclose(a, b, rel_tol, abs_tol):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

# check_symmetric checks to see if the numpy matrix a is symmetric.
def check_symmetric(a, tol = 1e-6):
  return numpy.allclose(a, a.T, atol = tol)


################################################################################
########################### Test RenegarIdentitySDP ############################

print "Testing RenegarIdentitySDP"

# Test on sample problem whose solution we know from using cvxpy.
A = [numpy.matrix([[1, 0, 1], [0, 3, 7], [1, 7, 5]]), numpy.matrix([[0, 2, 8], [2, 6, 0], [8, 0, 4]])]
c = numpy.matrix([[1, 2, 3], [2, 9, 0], [3, 0, 7]])
b = numpy.matrix([[6], [15]])
tol = 1e-5
max_iterations = 1000
eps = 0.1

run_times = []
iterations = []
objective_values = []

for i in numpy.linspace(-1,1,200):
    i = i * numpy.trace(c)
    start = timeit.default_timer()
    solution, solution_value, iteration_number = RenegarSDPv2(A,b,c,eps, tol, max_iterations, i)
    stop = timeit.default_timer()
    time = stop - start

    run_times.append(time)
    iterations.append(iteration_number)
    objective_values.append(solution_value)

    print "%d percent done", i / numpy.trace(c)

y = numpy.linspace(-1,1,200)*numpy.trace(c)
plt.scatter(y,run_times)
plt.title("Run Time as a function of z")
plt.xlabel("z value")
plt.ylabel("Run Time (seconds)")
plt.show()
plt.scatter(y, iterations)
plt.title("Iteration Number as a function of z")
plt.xlabel("z value")
plt.ylabel("Iteration Number")
plt.show()
plt.scatter(y,objective_values)
plt.title("Objective Value as a function of z")
plt.xlabel("z value")
plt.ylabel("Objective Value")
plt.show()

################################################################################
# Make plots for second example.
A = [numpy.matrix([[1, 2, 6], [2, 3, 7], [6, 7, 5]]), numpy.matrix([[0, 2, 8], [2, 6, 0], [8, 0, 4]])]
c = numpy.matrix([[1, 2, 3], [2, 9, 0], [3, 0, 7]])
b = numpy.matrix([[8], [4]])
eps = 0.1
tol = 1e-5
max_iterations = 1000


run_times = []
iterations = []
objective_values = []

for i in numpy.linspace(-1,1,200):
    i = i * numpy.trace(c)
    start = timeit.default_timer()
    solution, solution_value, iteration_number = RenegarSDPv2(A,b,c,eps, tol, max_iterations, i)
    stop = timeit.default_timer()
    time = stop - start

    run_times.append(time)
    iterations.append(iteration_number)
    objective_values.append(solution_value)

    print "%d percent done", i / numpy.trace(c)

y = numpy.linspace(-1,1,200)*numpy.trace(c)
plt.scatter(y,run_times)
plt.title("Run Time as a function of z")
plt.xlabel("z value")
plt.ylabel("Run Time (seconds)")
plt.show()
plt.scatter(y, iterations)
plt.title("Iteration Number as a function of z")
plt.xlabel("z value")
plt.ylabel("Iteration Number")
plt.show()
plt.scatter(y,objective_values)
plt.title("Objective Value as a function of z")
plt.xlabel("z value")
plt.ylabel("Objective Value")
plt.show()