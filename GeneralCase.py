import cvxpy
import numpy
import math
import IdentityCase


def IsPositiveSemidefinite(A):
	""" Input: A must be a numpy matrix.
		Output: True if A is positive semidefinite, False otherwise.
		Description: Checks to see if A is a positive semidefinite matrix
					 by seeing if it's a symmetric matrix with positive
					 eigenvalues.
	"""

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
	objective = cvxpy.Minimize(0)
	prob = cvxpy.Problem(objective, constraints)
	result = prob.solve()

	if (prob.status == 'infeasible'):
		raise ValueError('Problem Is Infeasible')

	if (prob.status == 'unbounded'):
		raise ValueError('Problem Is Unbounded')

	return x.value


def EigenDecomp(E):
	""" Input: (n x n)-dimensional numpy matrix E. E must be positive semidefinite.
		Output: Orthogonal matrix Q of normed eigenvectors and diagonal matrix
				of corresponding eigenvalues. Both given as (n x n)-dimensional
				numpy matrices.
		Description: Self-explanatory.
	"""

	eig_values, eig_vectors = numpy.linalg.eig(E)
	eig_values_matrix = numpy.diag(eig_values)
	return [eig_vectors, eig_values_matrix]


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
	[Q, eig_D] = EigenDecomp(E)
	eig_D_root = numpy.sqrt(eig_D)
	E_root = Q * eig_D_root * Q.T

	new_c = E_root * c * E_root
	new_A = []
	for i in range(0, len(A)):
		new_A.append(E_root * A[i] * E_root)

	opt_mod_sol, opt_mod_sol_val = IdentityCase.RenegarIdentitySDP(A, b, c, eps)
	opt_sol = E_root * opt_mod_sol * E_root

	return [opt_sol, opt_mod_sol_val]