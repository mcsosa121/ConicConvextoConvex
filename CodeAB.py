from cvxpy import *
from numpy import *

def Checker(A, b, c, epsilon):
    """ Checks to see if the user-defined input is of the correct form.
        We require that A be a list of m (n x n)-dimensional numpy matrices,
        b be an (m x 1)-dimensional numpy matrix, c be an (n x n)-dimensional
        numpy matrix, and epsilon be a number between 0 and 1 (not inclusive).
    """

    if (not isinstance(epsilon, float)) or (epsilon >= 1) or (epsilon <= 0):
        raise ValueError('epsilon must be a number between 0 and 1 (not inclusive).')

    if (not isinstance(c, matrix)) or (c.shape[0] != c.shape[1]):
        raise ValueError('c must be a square numpy matrix.')

    if (not isinstance(b, matrix)) or (b.shape[1] != 1):
        raise ValueError('b must be an (m x 1)-dimensional numpy matrix.')

    if (not isinstance(A, list)) or (len(A) == 0):
        raise ValueError('A must be a list that includes at least one constraint.')

    for i in range(0, len(A)):
        if (not isinstance(A[i], matrix)) or (A[i].shape[0] != A[i].shape[1]):
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


def InitialFinder(A, b, c):
    """ Precondition: A is a list of m (n x n)-dimensional numpy matrices, b is
                      an (m x 1)-dimensional numpy matrix, c is an
                      (n x n)-dimensional numpy matrix.
        Postcondition: Returns an (n x n)-dimensional numpy matrix that's
                       a feasible solution to the original problem CP (see page
                        3 in the paper).
        Description: cvxpy is used to find a feasible solution to the original
                     problem CP. The solution that's found will be used as a
                     distinguished direction.
    """

    n = c.shape[0]
    x = Variable(n, n)
    objective = Minimize(trace(c.T * x))
    constraints = [x >= 0]
    for i in range(0, len(A)):
        constraints += [trace(A[i].T * x) == b[i]]
    prob = Problem(objective, constraints)
    result = prob.solve()

    if (prob.status == 'infeasible'):
        raise ValueError('Problem Is Infeasible')

    if (prob.status == 'unbounded'):
        raise ValueError('Problem Is Unbounded')

    return x.value