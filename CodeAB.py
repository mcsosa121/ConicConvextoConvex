import cvxpy
from cvxpy import *

import numpy 
def Checker(A, b, c, epsilon):
    try: 
        n = numpy.shape(A)[0]
        Acol = numpy.shape(A)[1]
        Brow = numpy.shape(b)[0] 
        Bcol = numpy.shape(b)[1]
        Crow = numpy.shape(c)[0] 
        Ccol = numpy.shape(c)[1]
    except IndexError: 
        raise ValueError('A,b, and c must be 2-dimensional arrays or matrices')
    else: 
        if (Acol != n):
            raise ValueError('Matrix A must be square')
        elif (Brow != n or
              Bcol != n):
            raise ValueError('Matrix b must have same dimensions as A')
        elif (Crow != n or
              Ccol != n):
            raise ValueError('Matrix c must have same dimensions as A')
        elif (epsilon < 0 or epsilon >1):
            raise ValueError('epsilon must be between 0 and 1')
        elif(len(numpy.shape(A)) > 2 or 
            len(numpy.shape(b)) > 2 or 
            len(numpy.shape(c)) > 2):
            raise ValueError('Matrices should be no more than two dimensions')
        elif (numpy.isnan(A).any() or
             numpy.isnan(b).any() or 
             numpy.isnan(c).any()):
            raise ValueError('Matrices must not have any Nones or Infinities')
    return n

def InitialFinder(A,b,c,n):
    x = Variable(n,n)
    constr = [A*x ==b,
             trace(c*x) < trace(c)]
    prob = Problem(Minimize(0),constr)
    prob.solve()
    if (prob.status == 'infeasible'):
        raise ValueError('Problem is Infeasible')
    return x.value

p=5
A = numpy.zeros((p,p))
b = numpy.zeros((p,p))
c = numpy.zeros((p,p))
n = Checker(A,b,c,.5)
InitialFinder(A,b,c,n)