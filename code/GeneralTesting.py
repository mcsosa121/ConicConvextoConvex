from GeneralCase import *
import math
import numpy
import cvxpy
import IdentityCase


# isclose is used to check if two floating point numbers are close
# enough to be considered equal.
def isclose(a, b, rel_tol, abs_tol):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

# check_symmetric checks to see if the numpy matrix a is symmetric.
def check_symmetric(a, tol = 1e-6):
	return numpy.allclose(a, a.T, atol = tol)


################################################################################
######################### Test FindDistinguishedDirection ######################

print "Testing FindDistinguishedDirection"

# Example where given problem is infeasible.
A = [numpy.matrix([[1, 2], [3, 4]]), numpy.matrix([[1, 2], [3, 4]])]
b = numpy.matrix([[0], [5]])

try:
	x = FindDistinguishedDirection(A, b)
except:
	print "Checked that problem must be feasible."

# Example where given problem is feasible.
A = [numpy.matrix([[1, 0], [0, -1]])]
b = numpy.matrix([[0]])

x = FindDistinguishedDirection(A, b)

print "Checked that FindDistinguishedDirection works for correct input."

print "Finished testing FindDistinguishedDirection."


################################################################################
################################# Test EigenDecomp #############################

print "Testing EigenDecomp"
A = numpy.matrix([[2, 6], [6, 18]])
Q, D = EigenDecomp(A)

# Check that D is a diagonal matrix with all non-negative entries.
result_D = True

for i in range(0, D.shape[0]):
	for j in range(0, D.shape[0]):
		if D.item(i, j) <= -1e6:
			result_D = False

		if (i != j) and (not isclose(0, D.item(i, j), 1e-6, .01)):
			result_D = False

# Check that Q is an orthogonal matrix.
result_Q = True

for i in range(0, Q.shape[0]):
	if not isclose(1, numpy.linalg.norm(Q[:, i]), 1e-6, .01):
		result_Q = False

for i in range(0, Q.shape[0]):
	for j in range(i+1, Q.shape[0]):
		if not isclose(0, numpy.dot(Q[:, i].T, Q[:, j]), 1e-6, .01):
			result_Q = False

# Check that the diagonal entries of D and the columns of Q are
# eigenvalues and eigenvectors, respectively.
result_both = True

for i in range(0, Q.shape[0]):
	LHS = A * Q[:, i]
	RHS = D.item(i, i) * Q[:, i]
	for j in range(0, LHS.size):
		if not isclose(LHS[j], RHS[j], 1e-6, .01):
			result_both = False

if result_D and result_Q and result_both:
	print "Checked that EigenDecomp works correctly."
else:
	raise ValueError("EigenDecomp doesn't seem to work.")

print "Finished testing EigenDecomp"


################################################################################
################################# Test RenegarSDP ##############################

print "Testing RenegarSDP"

# Example where identity matrix is a feasible solution - should return same
# answer as RenegarIdentitySDP.

A = [numpy.matrix([[1, 0, 1], [0, 3, 7], [1, 7, 5]]), numpy.matrix([[0, 2, 8], [2, 6, 0], [8, 0, 4]])]
c = numpy.matrix([[1, 2, 3], [2, 9, 0], [3, 0, 7]])
b = numpy.matrix([[9], [10]])
eps = 0.1

gen_solution, gen_solution_value = RenegarSDP(A, b, c, eps)
true_solution_value = 8.92

if isclose(true_solution_value, gen_solution_value, 1e-1, 0.01):
	print "RenegarIdentitySDP seems to get correct value"
else:
	raise ValueError("Value returned by RenegarIdentitySDP function is not what we expect to see.")

eig_values, eig_vectors = numpy.linalg.eig(gen_solution)
smallest_eigenvalue = round(numpy.amin(eig_values), 6)

if isclose(numpy.trace(A[0].T * gen_solution), b[0], 1e-1, .01) and isclose(numpy.trace(A[1].T * gen_solution), b[1], 1e-1, .01) and check_symmetric(gen_solution) and smallest_eigenvalue >= 0:
	print "RenegarSDP gets a feasible solution."
else:
	raise ValueError("Solution returned by RenegarSDP function violates constraints i.e. isn't feasible.")

# Example where identity matrix is not a feasible solution.

A = [numpy.matrix([[1, 0, 1], [0, 3, 7], [1, 7, 5]]), numpy.matrix([[0, 2, 8], [2, 6, 0], [8, 0, 4]])]
c = numpy.matrix([[1, 2, 3], [2, 9, 0], [3, 0, 7]])
b = numpy.matrix([[6], [15]])
eps = 0.1

gen_solution, gen_solution_value = RenegarSDP(A, b, c, eps)
true_solution_value = 9.7668

if isclose(true_solution_value, gen_solution_value, 1e-1, 0.01):
	print "RenegarIdentitySDP seems to get correct value"
else:
	raise ValueError("Value returned by RenegarIdentitySDP function is not what we expect to see.")

eig_values, eig_vectors = numpy.linalg.eig(gen_solution)
smallest_eigenvalue = round(numpy.amin(eig_values), 6)

if isclose(numpy.trace(A[0].T * gen_solution), b[0], 1e-1, .01) and isclose(numpy.trace(A[1].T * gen_solution), b[1], 1e-1, .01) and check_symmetric(gen_solution) and smallest_eigenvalue >= 0:
	print "RenegarSDP gets a feasible solution."
else:
	raise ValueError("Solution returned by RenegarSDP function violates constraints i.e. isn't feasible.")

print "Finished testing RenegarSDP"