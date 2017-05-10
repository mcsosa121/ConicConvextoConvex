from GeneralCase import*
from GeneralCaseTolerance import*
import numpy as np
import cvxpy
import timeit as ti


def createRandGenSDP(m, n):
    """
    Creates a randomnly generated non-sparse SDP with uniform distribution.
    M is the number of A matrices you want to have
    n is the dimension that you want the A (nxn), b (nx1) and c (nxn)
      matrices to have
    """
    A = []
    for i in range(m):
        a = numpy.matrix(np.random.rand(n, n))
        a = a.T * a
        A.append(a)
    c = numpy.matrix(np.random.rand(n, n))
    c = c.T * c
    b = np.matrix(np.random.rand(m, 1))
    return [A, b, c]


def testCases(A, b, c):
    try:
        print "---------Starting Algorithm---------"
        start = ti.default_timer()
        sol, solvalue, iter = RenegarSDPv2(A, b, c, 0.03, 10e-5, 100000)
        stop = ti.default_timer()
        print "Finished in %f sec with %d iterations" % (stop - start, iter)
        print "---------Starting CVXpy-------------"
        n = A[0].shape[0]
        x = cvxpy.Semidef(n)
        constraints = []
        for i in range(0, len(A)):
            constraints += [cvxpy.trace(A[i].T * x) == b[i]]
        objective = cvxpy.Minimize(cvxpy.trace(c.T * x))
        prob = cvxpy.Problem(objective, constraints)
        result = prob.solve()
        print "Finished CVXpy"
        print "------------------------------------"
        print "CVXpy result: %f" % result
        print "Renegar Algorithm result: %f" % solvalue
        return True
    except ValueError:
        print "Error. Cases not optimal. Trying again"
        return False


print "Welcome to the Undergradpad SDP problem Generator"
m = None
n = None
while not isinstance(m, int):
    try:
        m = input("Please Enter the number of matrices you want A to have? : ")
        if m <= 0 or (not isinstance(m, int)):
            print "Error. Please enter a positive integer."
            m = None
    except NameError:
        print "Error not an Integer. Please enter a positive integer."


while not isinstance(n, int):
    try:
        n = input("Please enter the desired dimension of the matrices : ")
        if n <= 0 or (not isinstance(n, int)):
            print "Error. Please enter a positive integer"
            n = None
    except NameError:
        print "Error. Please enter a positive integer."

print "Running...."
result = False

while not result:
    A, b, c = createRandGenSDP(m, n)
    result = testCases(A, b, c)


print "Finished Generation. Here are your results"
print "A ------------------------------------"
ct = 0
for i in A:
    print "Matrix %d : " % ct
    ct += 1
    print i
print "b ------------------------------------"
print b
print "c ------------------------------------"
print c
