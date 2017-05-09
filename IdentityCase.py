import cvxpy
import numpy
import math

# isclose is used to check if two floating point numbers are close
# enough to be considered equal.
def isclose(a, b, rel_tol, abs_tol):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def Checker(A, b, c, epsilon):
    """ Checks to see if the user-defined input is of the correct form.
        We require that A be a list of m (nxn)-dimensional numpy matrices
        (with m >= 1), b be an (m x 1)-dimensional numpy matrix, c be an (n x n)-dimensional
        numpy matrix, and epsilon be a number between 0 and 1 (not inclusive).
    """
    
    if (not isinstance(epsilon, float)) or (epsilon >= 1) or (epsilon <= 0):
        raise ValueError('epsilon must be a number between 0 and 1 (not inclusive).')

    if (not isinstance(c, numpy.matrix)) or (c.shape[0] != c.shape[1]):
        raise ValueError('c must be a square numpy matrix.')

    if (not isinstance(b, numpy.matrix)) or (b.shape[1] != 1):
        raise ValueError('b must be an (m x 1)-dimensional numpy matrix.')

    if (not isinstance(A, list)) or (len(A) == 0):
        raise ValueError('A must be a list that includes at least one constraint.')

    for i in range(0, len(A)):
        if (not isinstance(A[i], numpy.matrix)) or (A[i].shape[0] != A[i].shape[1]):
            raise ValueError('Each element of A must be a square numpy matrix.')

    # Now we go through and make sure dimensions all match.
    n = c.shape[0]
    m = b.shape[0]

    if (len(A) != m):
        raise ValueError('Number of constraints in A should match the length of b.')

    for i in range(0, len(A)):
        if (A[i].shape[0] != n):
            raise ValueError('Dimensions of each matrix in A must match dimensions of c.')

    return [m, n]


def IsIdentitySolution(A, b):
    """ Input: A is a list of m (n x n)-dimensional numpy matrices, and b is 
               an (m x 1)-dimensional numpy matrix.
        Output: True or False
        Description: IsIdentitySolution returns True if the (n x n)-dimensional
                     identity matrix is a solution to <A_i, x> = b_i for each
                     A_i in A and b_i in b.
    """

    to_return = True
    n = A[0].shape[0]
    I = numpy.identity(n)

    for i in range(0, len(A)):
        if not isclose(numpy.trace(A[i].T * I), b[i], 1e-4, .01):
            to_return = False

    return to_return


def InitialFinder(A, b, c, z):
    """ Input: A is a list of m (n x n)-dimensional numpy matrices, b is
               an (m x 1)-dimensional numpy matrix, c is an
               (n x n)-dimensional numpy matrix, and z is a number that is
               less than the inner product of c and the identity matrix.
        Output: Returns an (n x n)-dimensional symmetric matrix that is in
                the affine space specified by the original problem CP (see page 3
                of paper) and whose objective function value is equal to z.
        Description: We use cvxpy to find a starting point for the supgradient algorithm
                     (see page 14 of paper).
    """
    
    n = c.shape[0]
    x = cvxpy.Variable(n, n)
    objective = cvxpy.Minimize(0)
    constraints = [cvxpy.trace(c.T * x) == z, x == x.T]
    for i in range(0, len(A)):
        constraints += [cvxpy.trace(A[i].T * x) == b[i]]
    prob = cvxpy.Problem(objective, constraints)
    result = prob.solve()
    
    if (prob.status == 'infeasible'):
        raise ValueError('Problem Is Infeasible')

    return x.value


def Supgradient(x):
    """ Input: x must be an (n x n)-dimensional numpy matrix that's a feasible
               solution to problem (2.5) for some valid value of z (see page 4 of paper).
               We're assuming that the distinguished direction e is the identity matrix.
        Output: Returns an (n x n)-dimensional numpy matrix that represents
                a supgradient of lambda_min(x) evaluated at the point x given.
        Description: Calculates the supgradient that will be used to update our
                     guess of the solution at each iteration of the main algorithm.
                     See page 5 of the paper for specifics. This function assumes that the
                     distinguished direction e is the identity matrix.
    """
    
    eig_values, eig_vectors = numpy.linalg.eig(x)
    sort_permutations = eig_values.argsort()
    
    # Sort eigenvalues and re-arrange eigenvectors to match up with eigenvalues.
    eig_values.sort()
    eig_vectors = eig_vectors[:, sort_permutations]
    
    v = eig_vectors[:, 0]
    return v * numpy.transpose(v)


def SupgradProjection(s, A, c):
    """ Input: s is an (n x n)-dimensional numpy matrix that represents a
               supgradient of lambda_min(x) for some x. A must be a list of
               (n x n)-dimensional numpy matrices that corresponds to the affine
               constraint in the original problem CP (see page 3 of paper),
               and c must be an (n x n)-dimensional numpy matrix that
               corresponds to the objective function in the original problem CP.
        Output: Returns an (n x n)-dimensional numpy matrix that's the
                projection of the supgradient s onto the intersection of the
                null spaces of the matrices in A and the matrix c.
        Description: Projects the supgradient s onto the intersection of the
                     null spaces of the matrices in A with the null space
                     of the matrix c (see page 5 of the paper).
    """

    sup_vec = numpy.matrix(numpy.ravel(s)).T
    c_vec = numpy.matrix(numpy.ravel(c))
    A_mat = numpy.matrix(numpy.ravel(A[0]))

    for i in range(1, len(A)):
        A_i = A[i]
        A_i_vec = numpy.matrix(numpy.ravel(A_i))
        A_mat = numpy.concatenate((A_mat, A_i_vec))

    A_mat = numpy.concatenate((A_mat, c_vec))

    # Compute projection matrix for null space of A_mat
    n_2 = len(sup_vec)
    n = int(math.sqrt(n_2))
    I = numpy.matrix(numpy.identity(n_2))
    P = I - A_mat.T * numpy.linalg.inv(A_mat * A_mat.T) * A_mat

    # Projected original supgradient onto null space of A_mat and reshape into
    # an (n x n)-dimensional matrix.
    new_sup = P * sup_vec

    to_return = new_sup[0:n].T
    for i in range(1, n):
        to_return = numpy.concatenate((to_return, new_sup[i*n:(i+1)*n].T))

    return to_return


def ModToOrig(x):
    """ Input: x is an (n x n)-dimensional numpy matrix that corresponds to a
               feasible solution to the modified problem (2.5) (see pages 3-4 of
               the paper).
        Output: An (n x n)-dimensional numpy matrix that corresponds to a
                feasible solution to the original problem, assuming that the
                distinguished direction used is the identity matrix.
        Description: ModToOrig maps the point x to the boundary of the feasible
                     region of the original problem. It corresponds to pi(x) in
                     the paper (see page 3). 
    """

    n = x.shape[0]
    e = numpy.identity(n)
    eig_values, eig_vectors = numpy.linalg.eig(x)
    eig_values_sorted = numpy.sort(eig_values)
    smallest_eig = eig_values_sorted[0]

    return e + (1 / (1 - smallest_eig)) * (x - e)


def AlgoMain(A, c, x, z, eps):
    """ Input: A must be a list of (n x n)-dimensional numpy matrices that
               correspond to the affine constraints in the original problem CP
               (see page 3), and c must be an (n x n)-dimensional numpy matrix
               that corresponds to the objective function in the original
               problem CP. x is an (n x n)-dimensional numpy matrix that's a
               feasible solution to the modified problem (see page 4). z is a
               number less than tr((c^T)I). Finally, eps is the desired
               relative accuracy of the solution and must be between 0 and 1
               (not inclusive).
        Output: An (n x n)-dimensional numpy matrix that represents the optimal
                solution to the original problem CP, along with this solution's
                value.
        Description: AlgoMain uses the supgradient projection algorithm described
                     on page 14 to find an approximate solution to the original
                     problem CP (see page 3). By "approximate solution" we mean
                     a solution pi* that satisfies (<c, pi*> - z*) / (<c, e> - z*) <= eps,
                     where z* is the true optimal value of the original problem CP.
    """

    # Initialize
    n = x.shape[0]
    e = numpy.identity(n)

    x_0 = ModToOrig(x)
    pi_0 = ModToOrig(x)

    x_k = [x_0]
    pi_k = [pi_0]

    # Compute Iteration Bound
    l = 1 / eps**3
    l = int(numpy.ceil(l))

    # Iterate
    for i in range(0, l):
        supgrad_k = Supgradient(x_k[i])
        project_supgrad_k = SupgradProjection(supgrad_k, A, c)
        x_tild = x_k[i] + (eps / 2) * (1 / numpy.linalg.norm(project_supgrad_k)) * project_supgrad_k
        pi_k.append(ModToOrig(x_tild))

        if numpy.trace(c.T * (e - pi_k[i+1])) >= (4/3) * numpy.trace(c.T * (e - x_tild)):
            x_k.append(pi_k[i+1])
        else:
            x_k.append(x_tild)

    opt_sol = pi_k[l]
    opt_sol_val = numpy.trace(c.T * opt_sol)

    return [opt_sol, opt_sol_val]


def RenegarIdentitySDP(A, b, c, eps, z=None):
    """ Input: A must be a list of (n x n)-dimensional numpy matrices that
               correspond to the affine constraints in the original problem CP
               (see page 3), and c must be an (n x n)-dimensional numpy matrix
               that corresponds to the objecive function in the original
               problem CP, and b is an (m x 1)-dimensional numpy matrix that
               represents the RHS of the affine constraints in CP. eps is a
               number that represents the desired accuracy of the returned
               solution relative to the actual solution. Furthermore, the identity
               matrix must be a solution to the SDP specified by A, b, and c.
        Output: A list with two elements: the first is an (n x n)-dimensional
                numpy matrix that's a solution to the original problem CP, and
                the second is the objective function value of this solution.
        Description: We use the algorithm described in Renegar's paper to solve
                     the following SDP: minimize <c, x> subject to Ax = b and
                     x is in the cone of positive semidefinite matrices.
    """

    # First, check to see that the input given is valid.
    m, n = Checker(A, b, c, eps)

    if not IsIdentitySolution(A, b):
        raise ValueError("Identity matrix is not a solution for this input.")

    # Now, find a starting point for the supgradient algorithm.
    n = c.shape[0]
    e_value = numpy.trace(c.T * numpy.identity(n))
    if z is None:
        z = e_value - .9 * abs(e_value)
    else:
        z = z
    initial_guess = InitialFinder(A, b, c, z)

    # Finally, run supgradient algorithm and return optimal solution along
    # with its value.

    opt_sol, opt_sol_val = AlgoMain(A, c, initial_guess, z, eps)

    return [opt_sol, opt_sol_val]