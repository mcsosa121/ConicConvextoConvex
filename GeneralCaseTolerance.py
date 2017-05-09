import GeneralCase
import IdentityCase
import numpy
import cvxpy
import math

def AlgoMainWithTolerance(A, c, x, z, eps, tol = 1e-6, max_iterations=1000):
    """ Input: Same as AlgoMain in IdentityCase.py, except now we have tol and max_iterations,
               both of which must be numbers.
        Output: Same as AlgoMain in IdentityCase.py
        Description: AlgoMainWithTolerance uses the supgradient projection
                     algorithm described on page 14 to find an approximate solution
                     to the original problem CP (see page 3). Instead of using the
                     epsilon term mentioned in the paper to determine how long to run
                     the algorithm for, AlgoMainWithTolerance terminates after max_iterations
                     iterations have been run or until the relative difference between the
                     objective value of successive iterations is less than the tolerance tol.
    """

    n = x.shape[0]
    # Initialize
    e = numpy.identity(n)
    x_0 = IdentityCase.ModToOrig(x)
    pi_0 = IdentityCase.ModToOrig(x)

    x_k = [x_0]
    pi_k = [pi_0]

    iteration_number = 0
    objective_values = []

    # Iterate
    for i in range(0, max_iterations):
        supgrad_k = IdentityCase.Supgradient(x_k[i])
        project_supgrad_k = IdentityCase.SupgradProjection(supgrad_k, A, c)
        x_tild = x_k[i] + (eps / 2) * (1 / numpy.linalg.norm(project_supgrad_k)) * project_supgrad_k
        pi_k.append(IdentityCase.ModToOrig(x_tild))

        if numpy.trace(c.T * (e - pi_k[i+1])) >= (4/3) * numpy.trace(c.T * (e - x_tild)):
            x_k.append(pi_k[i+1])
        else:
            x_k.append(x_tild)

        iteration_number += 1

        # Check how close successive 5-term averages are.
        current_value = numpy.trace(c.T * pi_k[i+1])
        objective_values.append(current_value)
        if ((i + 1) % 5) == 0:
            obj_len = len(objective_values)
            last_five_avg = sum(objective_values[-5:]) / 5
            last_last_five_avg = sum(objective_values[-10:-5]) / 5
            close_enough = abs(last_five_avg - last_last_five_avg) / abs(last_last_five_avg + .1e-12) <= tol

            if close_enough:
                break

    if iteration_number == max_iterations:
        print "Warning: Maximum number of iterations reached. Problem may be unbounded or the chosen tolerance was too small."

    opt_sol = pi_k[iteration_number]
    opt_sol_val = numpy.trace(c.T * opt_sol)

    return [opt_sol, opt_sol_val]


def RenegarIdentitySDPv2(A, b, c, eps, tol = 1e-6, max_iterations = 1000):
    """ Input: Same as RenegarIdentitySDPv2, except now we have tol and max_iterations,
               both of which must be numbers.
        Output: Same as RenegarIdentitySDP.
        Description: Same as RenegarIdentitySDP, except now we use AlgoMainWithTolerance
                     instead of AlgoMain.
    """

    # First, check to see that the input give is valid.
    m, n = IdentityCase.Checker(A, b, c, eps)
    if not IdentityCase.IsIdentitySolution(A, b):
        raise ValueError("Identity matrix is not a solution for this input.")

    # Now, find a starting point for the supgradient algorithm.
    n = c.shape[0]
    e_value = numpy.trace(c.T * numpy.identity(n))
    z = e_value - .9 * abs(e_value)
    initial_guess = IdentityCase.InitialFinder(A, b, c, z)

    # Finally, run supgradient algorithm and return optimal solution along
    # with its value.

    opt_sol, opt_sol_val = AlgoMainWithTolerance(A, c, initial_guess, z, eps, tol, max_iterations)

    return [opt_sol, opt_sol_val]


def RenegarSDPv2(A, b, c, eps, tol = 1e-6, max_iterations = 1000):
    """ Input: Same as RenegarSDP, except now we have tol and max_iterations,
               both of which must be numbers.
        Output: Same as RenegarSDP.
        Description: Same as RenegarSDP, except now we use RenegarIdentitySDPv2
                     instead of RenegarIdentitySDP.
    """

    m, n = IdentityCase.Checker(A, b, c, eps)
    E = GeneralCase.FindDistinguishedDirection(A, b)
    [Q, eig_D] = GeneralCase.EigenDecomp(E)
    eig_D_root = numpy.sqrt(eig_D)
    E_root = Q * eig_D_root * Q.T

    new_c = E_root * c * E_root
    new_A = []
    for i in range(0, len(A)):
        new_A.append(E_root * A[i] * E_root)

    opt_mod_sol, opt_mod_sol_val = RenegarIdentitySDPv2(new_A, b, new_c, eps, tol, max_iterations)
    opt_sol = E_root * opt_mod_sol * E_root

    return [opt_sol, opt_mod_sol_val]