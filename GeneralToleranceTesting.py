from GeneralCaseTolerance import *
from IdentityCase import *
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
############################ Test AlgoMainWithTolerance ########################

print "Testing AlgoMainWithTolerance"

# Test on sample problem whose solution we know from using cvxpy to solve it.
A = [numpy.matrix([[1, 0, 1], [0, 3, 7], [1, 7, 5]]), numpy.matrix([[0, 2, 8], [2, 6, 0], [8, 0, 4]])]
c = numpy.matrix([[1, 2, 3], [2, 9, 0], [3, 0, 7]])
b = numpy.matrix([[9], [10]])
z = 15
initial_guess = InitialFinder(A, b, c, z)
eps = 0.1
tol = 1e-3
max_iterations = 1000

# Used CVX in Matlab to find the value of the actual optimal solution.
true_solution_value = 8.92

AlgoMainSol, AlgoMainSolVal = AlgoMainWithTolerance(A, c, initial_guess, z, eps, tol, max_iterations)

if isclose(true_solution_value, AlgoMainSolVal, 1e-1, 0.1):
	print "AlgoMainWithTolerance seems to get correct value"
else:
	raise ValueError("Value returned by AlgoMainWithTolerance function is not what we expect to see")

eig_values, eig_vectors = numpy.linalg.eig(AlgoMainSol)
smallest_eigenvalue = round(numpy.amin(eig_values), 6)

if isclose(numpy.trace(A[0].T * AlgoMainSol), b[0], 1e-1, .01) and isclose(numpy.trace(A[1].T * AlgoMainSol), b[1], 1e-1, .01) and check_symmetric(AlgoMainSol) and smallest_eigenvalue >= 0:
	print "AlgoMainWithTolerance gets a feasible solution"
else:
	raise ValueError("Solution returned by AlgoMainWithTolerance function violates constraints i.e. isn't feasible")

print "Finished testing AlgoMainWithTolerance"


################################################################################
############################ Test RenegarIdentitySDPv2 #########################

print "Testing RenegarIdentitySDPv2"

# Test on sample problem whose solution we know from using cvxpy.
A = [numpy.matrix([[1, 0, 1], [0, 3, 7], [1, 7, 5]]), numpy.matrix([[0, 2, 8], [2, 6, 0], [8, 0, 4]])]
c = numpy.matrix([[1, 2, 3], [2, 9, 0], [3, 0, 7]])
b = numpy.matrix([[9], [10]])
eps = 0.1
tol = 1e-3
max_iterations = 1000

solution, solution_value = RenegarIdentitySDPv2(A, b, c, eps, tol, max_iterations)
true_solution_value = 8.92

if isclose(true_solution_value, solution_value, 1e-1, 0.01):
	print "RenegarIdentitySDPv2 seems to get correct value."
else:
	raise ValueError("Value returned by RenegarIdentitySDPv2 seems to get correct value.")

eig_values, eig_vectors = numpy.linalg.eig(solution)
smallest_eigenvalue = round(numpy.amin(eig_values), 6)

if isclose(numpy.trace(A[0].T * solution), b[0], 1e-1, .01) and isclose(numpy.trace(A[1].T * solution), b[1], 1e-1, .01) and check_symmetric(solution) and smallest_eigenvalue >= 0:
	print "RenegarIdentitySDPv2 gets a feasible solution"
else:
	raise ValueError("Solution returned by RenegarIdentitySDPv2 function violates constraints i.e. isn't feasible")

print "Finished testing RenegarIdentitySDPv2"


################################################################################
################################ Test RenegarSDPv2 #############################

print "Testing RenegarSDPv2"

# Example where identity matrix is a feasible solution - should return same
# answer as RenegarIdentitySDP.

A = [numpy.matrix([[1, 0, 1], [0, 3, 7], [1, 7, 5]]), numpy.matrix([[0, 2, 8], [2, 6, 0], [8, 0, 4]])]
c = numpy.matrix([[1, 2, 3], [2, 9, 0], [3, 0, 7]])
b = numpy.matrix([[9], [10]])
eps = 0.1
tol = 1e-2
max_iterations = 1000

gen_solution, gen_solution_value = RenegarSDPv2(A, b, c, eps, tol, max_iterations)
true_solution_value = 8.92
print gen_solution_value

if isclose(true_solution_value, gen_solution_value, 1e-2, 0.01):
	print "RenegarIdentitySDPv2 seems to get correct value"
else:
	raise ValueError("Value returned by RenegarIdentitySDPv2 function is not what we expect to see.")

eig_values, eig_vectors = numpy.linalg.eig(gen_solution)
smallest_eigenvalue = round(numpy.amin(eig_values), 6)

if isclose(numpy.trace(A[0].T * gen_solution), b[0], 1e-1, .01) and isclose(numpy.trace(A[1].T * gen_solution), b[1], 1e-1, .01) and check_symmetric(gen_solution) and smallest_eigenvalue >= 0:
	print "RenegarSDPv2 gets a feasible solution."
else:
	raise ValueError("Solution returned by RenegarSDPv2 function violates constraints i.e. isn't feasible.")

# Example where identity matrix is not a feasible solution.

A = [numpy.matrix([[1, 0, 1], [0, 3, 7], [1, 7, 5]]), numpy.matrix([[0, 2, 8], [2, 6, 0], [8, 0, 4]])]
c = numpy.matrix([[1, 2, 3], [2, 9, 0], [3, 0, 7]])
b = numpy.matrix([[6], [15]])
eps = 0.1
tol = 1e-2
max_iterations = 1000

gen_solution, gen_solution_value = RenegarSDPv2(A, b, c, eps, tol, max_iterations)
true_solution_value = 9.7668
print gen_solution_value

if isclose(true_solution_value, gen_solution_value, 1e-1, 0.01):
	print "RenegarIdentitySDPv2 seems to get correct value"
else:
	raise ValueError("Value returned by RenegarIdentitySDPv2 function is not what we expect to see.")

eig_values, eig_vectors = numpy.linalg.eig(gen_solution)
smallest_eigenvalue = round(numpy.amin(eig_values), 6)

if isclose(numpy.trace(A[0].T * gen_solution), b[0], 1e-1, .01) and isclose(numpy.trace(A[1].T * gen_solution), b[1], 1e-1, .01) and check_symmetric(gen_solution) and smallest_eigenvalue >= 0:
	print "RenegarSDPv2 gets a feasible solution."
else:
	raise ValueError("Solution returned by RenegarSDPv2 function violates constraints i.e. isn't feasible.")

print "Finished testing RenegarSDPv2"