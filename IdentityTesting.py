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
################################ Test Checker ##################################
 
print "Testing Checker"

# Example where epsilon is not a float
A = [numpy.matrix([[1, 2], [3, 4]])]
c = numpy.matrix([[1, 2], [3, 4]])
b = numpy.matrix([[1]])
eps = "epsilon"

try:
	m, n = Checker(A, b, c, eps)
except ValueError:
	print "Checked that eps must be a number"

# Example where epsilon is less than or equal to 0
eps = -0.0

try:
	m, n = Checker(A, b, c, eps)
except ValueError:
	print "Checked that eps must be greater than 0"

# Example where epsilon is greater than or equal to 1
eps = 1.0

try:
	m, n = Checker(A, b, c, eps)
except ValueError:
	print "Checked that eps must be less that 1"

# Example where c is not a matrix
eps = 0.5
c = 10

try:
	m, n = Checker(A, b, c, eps)
except ValueError:
	print "Checked that c must be a matrix"

# Example where c is not symmetric
c = numpy.matrix([[1, 2, 3], [4, 5, 6]])

try:
	m, n = Checker(A, b, c, eps)
except ValueError:
	print "Checked that c must be symmetric"

# Example where b is not a matrix
c = numpy.matrix([[1, 2], [3, 4]])
b = 10

try:
	m, n = Checker(A, b, c, eps)
except ValueError:
	print "Checked that b must be a matrix"

# Example where b has more than one column
b = numpy.matrix([[1, 2], [3, 4]])

try:
	m, n = Checker(A, b, c, eps)
except ValueError:
	print "Checked that b must be a column vector"

# Example where A is not a list
b = numpy.matrix([[1]])
A = numpy.matrix([[1, 2], [3, 4]])

try:
	m, n = Checker(A, b, c, eps)
except ValueError:
	print "Checked that A must be a list"

# Example where A is an empty list
A = []

try:
	m, n = Checker(A, b, c, eps)
except ValueError:
	print "Checked that A can't be an empty list"

# Example where elements of A aren't matrices
A = [10]

try:
	m, n = Checker(A, b, c, eps)
except ValueError:
	print "Checked that elements of A must be matrices"

# Example where elements of A aren't symmetric
A = [numpy.matrix([[1, 2, 3], [4, 5, 6]])]

try:
	m, n = Checker(A, b, c, eps)
except ValueError:
	print "Checked that elements of A must be symmetric matrices"

# Example where length of A doesn't equal number of entries in b
A = [numpy.matrix([[1, 2], [3, 4]]), numpy.matrix([[1, 2], [3, 4]])]

try:
	m, n = Checker(A, b, c, eps)
except ValueError:
	print "Checked that length of A must equal number of entries in b"

# Example where elements of A aren't all the same dimension of c
A = [numpy.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])]

try:
	m, n = Checker(A, b, c, eps)
except:
	print "Checked that dimension of matrices in A must have same dimension as c"

# Example where everything works out
A = [numpy.matrix([[1, 2], [3, 4]]), numpy.matrix([[1, 2], [3, 4]])]
c = numpy.matrix([[1, 2], [3, 4]])
b = numpy.matrix([[1], [2]])
eps = 0.5

m, n = Checker(A, b, c, eps)

print "Checked that everything works for correct input"

print "Finished testing Checker"


################################################################################
########################### Test IsIdentitySolution ############################

# Example where identity matrix is a feasible solution.

A = [numpy.matrix([[1, 0, 1], [0, 3, 7], [1, 7, 5]]), numpy.matrix([[0, 2, 8], [2, 6, 0], [8, 0, 4]])]
b = numpy.matrix([[9], [10]])

if IsIdentitySolution(A, b):
	print "IsIdentitySolution works for example where identity matrix is a feasible solution."
else:
	raise ValueError("IsIdentitySolution doesn't work for example where identity matrix is a feasible solution.")


# Example where identity matrix is not a feasible solution.

A = [numpy.matrix([[1, 0, 1], [0, 3, 7], [1, 7, 5]]), numpy.matrix([[0, 2, 8], [2, 6, 0], [8, 0, 4]])]
b = numpy.matrix([[10], [10]])

if not IsIdentitySolution(A, b):
	print "IsIdentitySolution works for example where identity matrix is not a feasible solution."
else:
	raise ValueError("IsIdentitySolution doesn't work for example where identity matrix is not a feasible solution.")

print "Finished testing IsIdentitySolution"


################################################################################
############################# Test InitialFinder ###############################

print "Testing InitialFinder"

# Example where problem is infeasible
A = [numpy.matrix([[1, 2], [3, 4]]), numpy.matrix([[1, 2], [3, 4]])]
b = numpy.matrix([[0], [5]])
c = numpy.matrix([[1, 2], [3, 4]])

try:
	x = InitialFinder(A, b, c, 5)
except:
	print "Checked that problem must be feasible"

# Example where problem is unbounded
A = [numpy.matrix([[1, 0], [0, -1]])]
b = numpy.matrix([[0]])
c = numpy.matrix([[-1, 0], [0, -1]])

try:
	x = InitialFinder(A, b, c, -2)
except:
	print "Checked that problem must be bouded"

# Example where problem has a feasible solution
A = [numpy.matrix([[1, 0], [0, -1]])]
b = numpy.matrix([[0]])
c = numpy.matrix([[1, 0], [0, 1]])

x = InitialFinder(A, b, c, 1.5)

# Make sure that x is symmetric, satisfies the affine constraint, 
# and has objective function value equal to 1.5.

if isclose(numpy.trace(A[0].T * x), b.item(0, 0), 1e-6, 0.01) and isclose(numpy.trace(c.T * x), 1.5, 1e-6, 0.01) and check_symmetric(x):
	print "Checked that InitialFinder works for correct input"
else:
	raise ValueError("x isn't actually a solution to the desired problem.")

print "Finished testing InitialFinder"



if isclose(numpy.trace(A[0].T * x), b.item(0, 0), 1e-6, 0.01) and numpy.trace(c.T * x) <= numpy.trace(c.T * numpy.identity(c.shape[0])):
	print "Checked that InitialFinder works for correct input"


################################################################################
############################## Test Supgradient ################################

print "Testing Supgradient"

# Assuming inputted x is a positive semi-definite matrix, check to make sure
# that matrix returned by Supgradient function is what we expect.
x = numpy.matrix([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
supgrad_to_check = Supgradient(x)
expected_supgrad = numpy.matrix([[.25, 1/(2*math.sqrt(2)), .25], [1/(2*math.sqrt(2)), .5, 1/(2*math.sqrt(2))], [.25, 1/(2*math.sqrt(2)), .25]])
result = True
for i in range(0, 3):
    for j in range(0, 3):
        if not isclose(supgrad_to_check.item((i, j)), expected_supgrad.item((i, j)), 1e-6, .01):
            result = False
if result:
    print "Supgradient function works for x with distinct eigvalues and eigenvectors"
else:
    raise ValueError("Value returned by supgrad_to_check is not supgradient lambda_min at x.")

# Example where x has repeated eigenvalues.
x = numpy.matrix([[7, 0, -3], [-9, -2, 3], [18, 0, -8]])
supgrad_to_check = Supgradient(x)
expected_supgrad = numpy.matrix([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
result = True
for i in range(0, 3):
    for j in range(0, 3):
        if not isclose(supgrad_to_check.item((i, j)), expected_supgrad.item((i, j)), 1e-6, .01):
            result = False
if result:
    print "Supgradient function works for x with repeated eigenvalues"
else:
    raise ValueError("Value returned by Supgrad function is not supgradient of lambda_min at x.")

print "Finished testing Supgradient"


################################################################################
########################### Test SupgradProjection #############################

print "Testing SupgradProjection"

# Testing for specific instance of problem.  
A = [numpy.matrix([[1, 1, 0], [1, -1, -1], [0, -1, 0]]), numpy.matrix([[1, 2, 0], [2, 1, 2], [0, -3, 1]])]
c = numpy.matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
supgradient = numpy.matrix([[.25, 1/(2*math.sqrt(2)), .25], [1/(2*math.sqrt(2)), .5, 1/(2*math.sqrt(2))], [.25, 1/(2*math.sqrt(2)), .25]])

projected_supgrad_to_check = SupgradProjection(supgradient, A, c)
expected_projected_supgrad = numpy.matrix([[0.27974, 0.02848, -.03326], [0.02848, 0.32714, -0.17412], [-.03326, 0.18367, 0.17844]])
result = True

for i in range(0, 3):
	for j in range(0, 3):
		if not isclose(projected_supgrad_to_check.item((i, j)), expected_projected_supgrad.item((i, j)), 1e-6, .01):
			result = False

if result:
	print "Supgradient projection seems to work"
else:
	raise ValueError("Value returned by SupgradProjection function is not what we expect to see")

print "Finished testing SupgradProjection"


################################################################################
################################ Test AlgoMain #################################

print "Testing AlgoMain"

# Test on sample problem whose solution we know from using cvxpy to solve it.
A = [numpy.matrix([[1, 0, 1], [0, 3, 7], [1, 7, 5]]), numpy.matrix([[0, 2, 8], [2, 6, 0], [8, 0, 4]])]
c = numpy.matrix([[1, 2, 3], [2, 9, 0], [3, 0, 7]])
b = numpy.matrix([[9], [10]])
z = 15
initial_guess = InitialFinder(A, b, c, z)
eps = 0.1

# Used CVX in Matlab to find the value of the actual optimal solution.
true_solution_value = 8.92

AlgoMainSol, AlgoMainSolVal = AlgoMain(A, c, initial_guess, z, .1)

if isclose(true_solution_value, AlgoMainSolVal, 1e-1, 0.01):
	print "AlgoMain seems to get correct value"
else:
	raise ValueError("Value returned by AlgoMain function is not what we expect to see")

eig_values, eig_vectors = numpy.linalg.eig(AlgoMainSol)
smallest_eigenvalue = round(numpy.amin(eig_values), 6)

if isclose(numpy.trace(A[0].T * AlgoMainSol), b[0], 1e-1, .01) and isclose(numpy.trace(A[1].T * AlgoMainSol), b[1], 1e-1, .01) and check_symmetric(AlgoMainSol) and smallest_eigenvalue >= 0:
	print "AlgoMain gets a feasible solution"
else:
	raise ValueError("Solution returned by AlgoMain function violates constraints i.e. isn't feasible")

print "Finished testing AlgoMain"


################################################################################
########################### Test RenegarIdentitySDP ############################

print "Testing RenegarIdentitySDP"

# Test on sample problem whose solution we know from using CVX to solve in Matlab.
A = [numpy.matrix([[1, 0, 1], [0, 3, 7], [1, 7, 5]]), numpy.matrix([[0, 2, 8], [2, 6, 0], [8, 0, 4]])]
c = numpy.matrix([[1, 2, 3], [2, 9, 0], [3, 0, 7]])
b = numpy.matrix([[9], [10]])
eps = 0.1

solution, solution_value = RenegarIdentitySDP(A, b, c, eps)
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