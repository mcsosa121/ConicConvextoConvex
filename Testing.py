from CodeAB import *
from CodeThreeFiveSixSeven import *
from numpy import *
from cvxpy import *


################################################################################
################################ Test Checker ##################################
 
print "Testing Checker"

# Example where epsilon is not a float
A = [matrix([[1, 2], [3, 4]])]
c = matrix([[1, 2], [3, 4]])
b = matrix([[1]])
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
c = matrix([[1, 2, 3], [4, 5, 6]])

try:
	m, n = Checker(A, b, c, eps)
except ValueError:
	print "Checked that c must be symmetric"

# Example where b is not a matrix
c = matrix([[1, 2], [3, 4]])
b = 10

try:
	m, n = Checker(A, b, c, eps)
except ValueError:
	print "Checked that b must be a matrix"

# Example where b has more than one column
b = matrix([[1, 2], [3, 4]])

try:
	m, n = Checker(A, b, c, eps)
except ValueError:
	print "Checked that b must be a column vector"

# Example where A is not a list
b = matrix([[1]])
A = matrix([[1, 2], [3, 4]])

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
A = [matrix([[1, 2, 3], [4, 5, 6]])]

try:
	m, n = Checker(A, b, c, eps)
except ValueError:
	print "Checked that elements of A must be symmetric matrices"

# Example where length of A doesn't equal number of entries in b
A = [matrix([[1, 2], [3, 4]]), matrix([[1, 2], [3, 4]])]

try:
	m, n = Checker(A, b, c, eps)
except ValueError:
	print "Checked that length of A must equal number of entries in b"

# Example where elements of A aren't all the same dimension of c
A = [matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])]

try:
	m, n = Checker(A, b, c, eps)
except:
	print "Checked that dimension of matrices in A must have same dimension as c"

# Example where everything works out
A = [matrix([[1, 2], [3, 4]]), matrix([[1, 2], [3, 4]])]
c = matrix([[1, 2], [3, 4]])
b = matrix([[1], [2]])
eps = 0.5

m, n = Checker(A, b, c, eps)

print "Checked that everything works for correct input"

print "Finished testing Checker"


################################################################################
############################# Test InitialFinder ###############################

print "Testing InitialFinder"

# Example where problem is infeasible
A = [matrix([[1, 2], [3, 4]]), matrix([[1, 2], [3, 4]])]
b = matrix([[0], [5]])
c = matrix([[1, 2], [3, 4]])

try:
	x = InitialFinder(A, b, c)
except:
	print "Checked that problem must be feasible"

# Example where problem is unbounded
A = [matrix([[1, 0], [0, -1]])]
b = matrix([[0]])
c = matrix([[-1, 0], [0, -1]])

try:
	x = InitialFinder(A, b, c)
except:
	print "Checked that problem must be bouded"

# Example where problem has a feasible solution
A = [matrix([[1, 2], [3, 4]])]
b = matrix([[0]])
c = matrix([[1, 2]])

x = InitialFinder(A, b, c)
print "Checked that everything works for correct input"

print "Finished testing InitialFinder"


################################################################################
############################## Test Supgradient ################################

print "Testing Supgradient"


################################################################################
########################### Test SupgradProjection #############################

print "Testing SupgradProjection"


################################################################################
############################### Test ModToOrig #################################

print "Testing ModToOrig"

################################################################################
################################ Test AlgoMain #################################

print "Testing AlgoMain"