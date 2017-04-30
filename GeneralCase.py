import cvxpy
import numpy
import math
import IdentityCase



# Order of things to do:
# 1) Check that input is in the correct form.
#    	Use Checker from IdentityCase
#		Done
# 2) Find a feasible solution E to the given SDP.
#		Done
# 3) Compute Eigendecomposition of E.
# 4) Compute square root of E.
# 5) Compute new objective matrix C' from C and new affine matrices A_i' from A_i.
# 6) Solve modified problem with IdentityCase
# 7) Convert back to solution of the original problem.




def FindDistinguishedDirection(A, b):
	""" Input: A must be a list of m (n x n)-dimensional numpy matrices and b
			   b must be an (m x 1)-dimensional numpy matrix.
		Output: An (n x n)-dimensional numpy matrix that represents a feasible
				solution to the SDP specified by the affine constraints using
				A and b.
		Description: Use cvxpy to find a feasible solution to the SDP specified
					 by A and b.
	"""

	n = A[0].shape[0]
	x = cvxpy.Variable(n, n)
	constraints = [x >> 0, x.T == x]
	for i in range(0, len(A)):
		constraints += [cvxpy.trace(A[i].T * x) == b[i]]
	objective = 0
	prob = cvxpy.Problem(objective, constraints)
	result = prob.solve()

	if (prob.status == 'infeasible'):
        raise ValueError('Problem Is Infeasible')
    
    if (prob.status == 'unbounded'):
        raise ValueError('Problem Is Unbounded')

    return x.value


def EigenDecomp(E):
	""" Input: 
		Output: Orthogonal matrix Q of normed eigenvectors and diagonal matrix
				of corresponding eigenvalues. Both given as (n x n)-dimensional
				numpy matrices.
		Description: Self-explanatory. Eigenvalues and eigenvectors are sorted
					 in increasing order.
	"""


def RenegarSDP(A, b, c, eps):
	""" Input: A must be a list of m (n x n)-dimensional numpy matrices that
			   correspond to the affine constraints in the original problem CP
			   (see page 3), and c must be an (n x n)-dimensional numpy matrix
			   that corresponds to the objective function in the original
			   problem CP, and b is an (m x 1)-dimensional numpy matrix that
			   represents the RHS of the affine constraints in CP. eps is a
			   number that represents the desired accuracy of the returned
			   solution relative to the actual solution.
		Output: A list with two elements: the first is an (n x n)-dimensional
				numpy matrix that's a solution to the original problem CP, and
				the second is the objective function value of this solution.
		Description: We use the algorithm described in Renegar's paper to solve
					 the following SDP: minimize <c, x> subject to Ax = b and
					 x is in the cone of positive semidefinite matrices.
	"""

	m, n = IdentityCase.Checker(A, b, c, eps)

	E = FindDistinguishedDirection(A, b)