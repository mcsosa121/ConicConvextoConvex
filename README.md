# The Undergradpad
Implementing master of Optimization and Cornell Professor Jim Renegar's frameworks for applying Subgradient Method to Conic Optimization problems. 
Read Renegars original paper [here].
Read our [final report].
# The Team
* Zach Rosenof : (zer2)
* Mitch Perry : (mop25)
* [Mike Sosa] : (mcs348)

# Code Overview
1. GeneralCase
	* FindDistinguisedDirection - Use cvxpy to find a feasible solution to the SDP specified by A and b.
	    * Inputs :
	        1. A : list of numpy matrices - A list of m finite dimensional NxN matrices  
	        2. b : numpy matrix - Mx1 matrix
	   * Output :
	        1. e : numpy matrix - A NxN matrix that can serve as a distinguished direction to the problem
	* RenegarSDP - Uses the algorithm described in Renegar's paper to solve the following SDP: minimize <c, x> subject to Ax=                     b and x is in the cone of positive semidefinite matrices.
	   * Inputs :
	        1. A : list of numpy matrices - A list of m finite dimensional NxN matrices  
	        2. b : numpy matrix - Mx1 matrix
	        3. c : numpy matrix - NxN matrix
	        4. eps : float - Desired accuracy of solution. Higher epsilon corresponds to higher computation time
	   * Output :
	        1. Sol : numpy matrix - A NxN matrix containing the solution to the original SDP
	        2. SolValue : float - The objective function value
2. IdentityCase
    * Checker - Checks to see if the user-defined input is of the correct form.
	   * Inputs :
	        1. A : list of numpy matrices - A list of m finite dimensional NxN matrices  
	        2. b : numpy matrix - Mx1 matrix
	        3. c : numpy matrix - NxN matrix
	        4. eps : float - Desired accuracy of solution. Between 0 and 1 
	   * Output :
	        1. m : int - The number of matrices within A. Only returned if tests pass
	        2. n : int - The dimension of the matrices. Only returned if tests pass
    * Supgradient - Calculates a supgradient that is used to update algorithm guess
        * Inputs :
            1. x : numpy matrix - A NxN matrix that is a feasible solution to the problem. 
        * Output :
            1. sup : numpy matrix - NxN matrix that is the supgradient of lambda_min(x) evaluated at the point x
    * SupgradProjection - Projects the supgradient s onto the intersection of the null spaces of the matrices in A with the null space of the matrix c
        * Inputs :
            1. s : numpy matrix - A NxN dimensional matrix that represents a supgradient of lambdamin for some x
            2. A : list of numpy matrices - A list of m finite dimensional NxN matrices  
            3. c : numpy matrix - NxN matrix
        * Output :
            1. proj : numpy matrix - NxN matrix that is the projection
3. GeneralCaseTolerance
    * RenegarSDPv2 - Same as RenegarSDP except with options for iterations, z value and tolerance
        * Inputs :
            1. A : list of numpy matrices - A list of m finite dimensional NxN matrices  
	        2. b : numpy matrix - Mx1 matrix
	        3. c : numpy matrix - NxN matrix
	        4. eps : float - Desired accuracy of solution. Higher epsilon corresponds to higher computation time
	        5. Tol : float - The desired tolerance of the solution
	        6. max_iterations : int - The max number of iterations the algorithm should run before stopping
	        7. z : float - Desired z value for the algorithm
        * Output :
            1. Sol : numpy matrix - A NxN matrix containing the solution to the original SDP
	        2. SolValue : float - The objective function value
	        3. iteration_number : int - The number of iterations the algorithm ran for
4. Reggen
    * File that can be run to generate and solve a random SDP problem. Prompts the user for the desired number of matrices within A, and desired dimension of the matrices
    * createRandGenSDP - Creates a randomnly generated non-sparse SDP with uniform distribution. 
        * Inputs :
            1. m : int - Desired number of A matrices
            2. n : int - Desired dimension of the matrices
        * Outputs :
            1. A : numpy matrix - Random NxN symmetric positive definite matrix
            2. b : numpy matrix - Random Mx1 matrix
            3. c : numpy matrix - Random NxN symmetric positive definite matrix
5. Testing
	* Identity Testing - Tests case where the identity is the distinguished direction.
	* General Testing - Tests general cases
	* General Tolerance Testing - Tests the solver without a fixed number of iterations.

[//]: #
[here]: <https://arxiv.org/pdf/1503.02611.pdf>
[Mike Sosa]: <http://www.github.com/mcsosa121>
[final report]: reports/6326FinalReport.pdf
