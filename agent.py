import math
from copy import copy, deepcopy
from grid import *

###############################################################################
#                               Agent Class                                   #
###############################################################################


class Agent:
    def __init__(self, beta, discount_factor=0.99):
        """
        Parameters
        ----------
        beta : float
            Parameter for stochastic setting.
        discount_factor : float, optional
            Parameter for the discount factor. Corresponds to gamma in theory
        """
        self.__grid = Grid()
        self.__beta = beta
        self.__gamma = discount_factor

    def simpleUpPolicyDisplay(self, N=6):
        """
        Move and display the agent  following the policy of always going up in
        the grid. Corresponds to the Q2 of the project.

        Parameters
        ----------
        N : int, optional
            Number of iterations wanted
        """

        # Printing the initial grid
        print("Initial grid : ")
        self.__grid.print()

        sum_rewards = 0

        # N iterations
        for i in range(1, N+1):
            print("Iteration {}:".format(i))
            # Moving the agent up and printing the new corresponding grid
            self.__grid.moveAgent(UP, self.__beta)
            self.__grid.print()
            sum_rewards += self.__grid.getRewardAgentPos()
            print("Current sum of rewards : {}".format(sum_rewards))
            print("-----")

        print("The display stopped after {} iterations.".format(N))

    def computeJ(self, policy, N=3, display=True):
        """
        Compute and display JN(x) from the initial grid. Corresponds to the
        Q3 of the project.

        Parameters
        ----------
        policy : list
            Stationary policy the agent will follow
        N : int, optional
            Number of iterations wanted
        display : bool, optional
            Bool to display or not the result

        Returns
        -------
        list
            J_mu^N
        """
        # To optimize the computation, the matrix containing Jn-1 can be safed.
        # `previous_Jmatrix` corresponds to this Jn-1 matrix. It is instiated
        # with 0 as J0(x) = 0 for all x.
        previous_Jmatrix = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

        # `current_Jmatrix` corresponds to the matrix containing the Jn values
        current_Jmatrix = deepcopy(previous_Jmatrix)

        # N iterations
        for n in range(1, N+1):
            # Going over the grid from top left to bottom right
            for y in range(self.__grid.height):
                for x in range(self.__grid.width):
                    # Computing the index corresponding to the 1D matrix
                    index = y * self.__grid.width + x
                    # Getting the direction from the policy
                    dir = policy[index]
                    # Computing`previous_J` and `previous_J_beta` which
                    # corresponds to Jn-1(f(x, µ(x), w)) from the definition
                    previous_J = getFromMatrix(dir, previous_Jmatrix, x, y)
                    previous_J_beta = previous_Jmatrix[0][0]
                    # first_term corresponds to the deterministic case
                    first_term = (1 - self.__beta) * (
                        getFromMatrix(dir, self.__grid.matrix, x, y) +
                        self.__gamma * previous_J)
                    # second_term corresponds to the stochastic term
                    second_term = self.__beta * (
                        getFromMatrix(UP, self.__grid.matrix, 0, 0) +
                        self.__gamma * previous_J_beta)
                    # `Jx` is the final result and corresponds to Jn(x)
                    Jx = first_term + second_term
                    # Jn(x) is safed to be reused in futur computation
                    current_Jmatrix[y][x] = Jx
                    # Printing the result in a "array way"
                    if n == N and display:
                        print(str(round(Jx, 2)).rjust(8), end=' ')
                # Line break in the array display
                if n == N and display:
                    print(" ")
            # Preparing next iterations by setting the matrix Jn as Jn-1
            previous_Jmatrix = deepcopy(current_Jmatrix)
        # Returning last J computed
        return current_Jmatrix

    def getSimplePolicy(self, dir):
        """
        Generate a simple policy of always going to the same direction

        Parameters
        ----------
        dir : {0, 1, 2, 3}
            Direction wanted.

        Returns
        -------
        list[{0, 1, 2, 3}]
            The corresponding stationary policy
        """
        policy = []
        for i in range(len(self.__grid.matrix1D)):
            policy.append(dir)
        return policy
