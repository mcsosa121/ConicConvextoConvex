from IdentityCase import*
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
b = numpy.matrix([[9], [10]])
eps = 0.1

optz = None
opttime = None
time = 0
for i in numpy.linspace(-1,1,200):
    i = i * numpy.trace(c)
    start = timeit.default_timer()
    solution, solution_value = RenegarIdentitySDP(A,b,c,eps,i)
    stop = timeit.default_timer()
    time = stop - start
    if optz == None:
        optz = i
        opttime = time
        print "%d percent done",i
    elif time < opttime:
        optz = i
        opttime = time
        print "%d percent done",i
        print "New opttime"
    else:
        print "%d percent done",i

print "Optimum Z: " + str(optz)
print "Optimum time: " + str(opttime)

#making sure the solution works again.
true_solution_value = 8.92

if isclose(true_solution_value, solution_value, 1e-1, 0.01):
	print "RenegarIdentitySDP seems to get correct value"
else:
	raise ValueError("Value returned by RenegarIdentitySDP function is not what we expect to see")

eig_values, eig_vectors = numpy.linalg.eig(solution)
smallest_eigenvalue = round(numpy.amin(eig_values), 6)

if isclose(numpy.trace(A[0].T * solution), b[0], 1e-1, .01) and isclose(numpy.trace(A[1].T * solution), b[1], 1e-1, .01) and check_symmetric(solution) and smallest_eigenvalue >= 0:
	print "RenegarIdentitySDP gets a feasible solution"
else:
	raise ValueError("Solution returned by RenegarIdentitySDP function violates constraints i.e. isn't feasible")

print "Finished testing RenegarIdentitySDP"
