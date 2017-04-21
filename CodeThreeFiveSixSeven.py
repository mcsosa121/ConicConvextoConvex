from numpy import *
from math import *

def Supgradient(x):
	""" Input: x must be an (n x n)-dimensional numpy matrix that's a feasible
			   solution to problem (2.5) (see page 4 of paper).
		Output: Returns an (n x 1)-dimensional numpy matrix that represents
				a supgradient of lambda_min(x) evaluated at the point x.
		Description: Calculates the supgradient that will be used to update our
					 guess of the solution at each iteration. See page 5 of the
					 paper for specifics.
	"""
	
	eig_values, eig_vectors = linalg.eig(x)
	sort_permutations = eig_vals.argsort()

	# Sort eigenvalues and re-arrange eigenvectors to match up with eigenvalues.
	eig_values.sort()
	eig_vectors = eig_vectors[:, sort_permutations]

	smallest_eig_vector = eig_vectors[0]
	v = smallest_eig_vector / linalg.norm(smallest_eig_vector)
	return v * transpose(v)


def SupgradProjection(s, A, c):
	""" Input: s is an (n x n)-dimensional numpy matrix that represents a
			   supgradient of lambda_min(x). A must be a list of
			   (n x n)-dimensional numpy matrices and corresponds to the affine
			   constraint in the original problem CP (see page 3 of the paper),
			   and c must be an (n x n)-dimensional numpy matrix that
			   corresponds to the objective fuction in the original problem CP.
		Output: Returns an (n x n)-dimensional numpy matrix that's the
				projection of the supgradient s onto a trnslation of the
				intersection of the null spaces of the matrices in A and the
				line tr((c^T)I) = z.
		Description: Projects the supgradient s onto the intersection of the
					 of the null spaces of the matrices in A with the null
					 space of the matrix c.
	"""

	# Reshape supgradient into a column vector and c into a row vector.
	sup_vec = matrix(ravel(s)).T
	c_vec = matrix(ravel(c))
	A_mat = c_vec

	# Reshape each linear constraint into a row vector and add to A_mat
	for i in range(0, len(A)):
		A_i = A[i]
		A_i_vec = matrix(ravel(A_i))
		A_mat = vstack([A_mat, A_i_vec])

	# Compute projection matrix for null space of A_mat
	n_2 = len(sup_vec)
	n = sqrt(n_2)
	I = matrix(identity(n_2))
	P = I - A_mat.T * linalg.inv(A_mat * A_mat.T) * A_mat

	# Project original supgradient onto null space of A_mat and reshape into
	# an (n x n)-dimensional matrix.
	new_sup = P * sup_vec

	to_return = new_sup[0:n].T
	for i in range(1, n):
		to_return = vstack([to_return, new_sup[i*n, (i+1)*n].T])

	return to_return


def ModToOrig(x, e):
	""" Input: x is an (n x n)-dimensional numpy matrix that corresponds to a
			   feasible solution to the original problem CP. e is an
			   (n x n)-dimensional numpy matrix that corresponds to a striclty
			   feasible soluution to the original problem.
		Output: An (n x n)-dimensional numpy matrix that corresponds to a
			    feasible solution to the original problem.
		Description: ModToOrig maps the point x to the boundary of the feasible
					 region of the original problem. It corresponds to pi(x) in
					 the paper (see page 3).
	"""

	eig_values, eig_vectors = linalg.eig(x)
	eig_values_sorted = sort(eig_values)
	smallest_eig = eig_values_sorted[0]

	return e + (1 / (1 - smallest_eig)) * (x - e)


def AlgoMain(A, c, x, z, e, eps):
	""" Input: A must be a list of (n x n)-dimensional numpy matrices that
			   correspond to the affine constraints in the original problem
			   CP (see page 3), and c must be an (n x n)-dimensional numpy
			   matrix that corresponds to the objective function in the original
			   problem CP. x is an (n x n)-dimensional numpy matrix that's a
			   feasible solution to the original problem CP. z is a value
			   less than tr((c^T)I). e is an (n x n)-dimensional numpy matrix
			   that corresponds to a striclty feasible solution to the original
			   solution CP. Finally, eps is the desired relative accuracy of the
			   solution and must be between 0 and 1 (not inclusive).
		Output: An (n x n)-matrix that represents the optimal solution to the
				original problem CP, along with this solution's value.
		Description: AlgoMain uses the algorithm described on page 14 to find an
					 approximate solution to the original problem CP. By
					 "approximate solution" we mean a solution pi* that
					 satisfies (<c, pi*> - z*) / (<c, e> - z*) <= eps, where z*
					 is the true optimal value of the original problem CP.
	"""

	# Initialize	
	x_0 = ModToOrig(x, e)
	pi_0 = ModToOrig(x, e)

	x_k = array([x_0])
	pi_k = array([pi_0])

	# Compute Iteration Bound
	l = 1 / eps^3

	# Iterate
	for i in range(0, l):
		supgrad_k = Supgradient(x_k[i])
		project_supgrad_k = SupgradProjection(supgrad_k, A, c, x_k[i], z)
		x_tild = x_k[i] + (eps / 2) * (1 / linalg.norm(project_supgrad_k)) * project_supgrad_k
		pi_k = append(pi_k, ModToOrig(x_tild, e))

		if matrix.trace(transpose(c) * (e - pi_k[i+1])) >= (4/3) * matrix.trace(transpose(c) * (e - x_tild)):
			x_k = append(x_k, pi_k[i+1])
		else:
			x_k = append(x_k, x_tild)

	opt_sol = pi_k[l]
	opt_sol_val = matrix.trace(transpose(c) * opt_sol)

	return numpy.array([opt_sol, opt_sol_val])