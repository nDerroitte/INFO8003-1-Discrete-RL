import random

###############################################################################
#                               Constants                                     #
###############################################################################
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
NUMBER_ACTIONS = 4
###############################################################################
#                               Grid Class                                    #
###############################################################################


class Grid:
    def __init__(self):
        # Initial reward grid
        self.matrix = [[-3, 1, -5, 0, 19], [6, 3, 8, 9, 10], [5, -8, 4, 1, -8],
                       [6, -9, 4, 19, -5], [-20, -17, -4, -3, 9]]
        self.matrix1D = [-3, 1, -5, 0, 19, 6, 3, 8, 9, 10, 5, -8, 4, 1, -8, 6,
                         -9, 4, 19, -5, -20, -17, -4, -3, 9]
        self.width = len(self.matrix[0])
        self.height = len(self.matrix)
        self.__init_pos = [0, 3]
        self.__agent_pos = self.__init_pos

    def print(self):
        """
        Display the reward grid where the agent is located by a *
        """
        for y in range(self.height):
            for x in range(self.width):
                # Case where the agent is located on the current cell
                if [x, y] == self.__agent_pos:
                    print((str(self.matrix[y][x])+"*").rjust(4), end='')
                else:
                    print(str(self.matrix[y][x]).rjust(4), end='')
            # Line break
            print(" ")

    def getRewardAgentPos(self):
        return self.matrix[self.__agent_pos[1]][self.__agent_pos[0]]

    def getReward(self, x, y):
        return self.matrix[y][x]

    def getRewardIndex(self, index):
        x = index % self.width
        y = index // self.width
        return self.getReward(x, y)

    def moveAgent(self, direction, beta):
        """
        Move the agent in the grid

        Parameters
        ----------
        direction : {0, 1, 2, 3}
            Direction wanted. 0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT
        beta : float
            Parameter for stochastic setting.
        """
        # Stochastic case
        if random.random() <= beta:
            self.__agent_pos = [0, 0]
            return
        # Otherwise, move to the direction if it is allowed
        if direction == UP and self.__agent_pos[1] > 0:
            self.__agent_pos[1] -= 1
        elif direction == RIGHT and self.__agent_pos[0] < self.width-1:
            self.__agent_pos[0] += 1
        elif direction == DOWN and self.__agent_pos[1] < self.height-1:
            self.__agent_pos[1] += 1
        elif direction == LEFT and self.__agent_pos[0] > 0:
            self.__agent_pos[0] -= 1

###############################################################################
#                               Functions                                     #
###############################################################################


def getFromMatrix(dir, matrix, x, y):
    """
    Get the cell next to [x][y] in the direction `dir` in a matrix `matrix`

    Parameters
    ----------
    dir : {0, 1, 2, 3}
        Direction wanted. 0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT
    matrix : matrix of float
        Matrix from where the cells are selected
    x, y : int
        Position of the initial cell in the matrix

    Returns
    -------
    float
        The cell next to [x][y] in the direction `dir in the matrix
        `matrix`
    """

    # {UP, RIGHT, DOWN, LEFT} are constants fixed in grid.py
    if dir == UP and y > 0:
        return matrix[y-1][x]
    elif dir == RIGHT and x < len(matrix[0])-1:
        return matrix[y][x+1]
    elif dir == DOWN and y < len(matrix)-1:
        return matrix[y+1][x]
    elif dir == LEFT and x > 0:
        return matrix[y][x-1]
    else:
        # Happen in the case where to agent is located next to a wall and
        # makes a move toward this wall
        return matrix[y][x]


def getIndexFromMatrix(dir, matrix, index):
    """
    Get the cell next in the direction dir from the index in the matrix.
    The trick is that the matrix is in 2D and the index in consider a 1D
    matrix.

    Parameters
    ----------
    dir : {0, 1, 2, 3}
        Direction wanted. 0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT
    matrix : matrix of float
        Matrix from where the cells are selected
    index : int
        Position of the initial cell in the matrix

    Returns
    -------
    int
        Index of the cell next to the starting positon `index` in the direction
        `dir in the matrix `matrix`
    """
    # Computing corresponding x, y
    x = index % len(matrix[0])
    y = index // len(matrix[0])
    if dir == UP:
        return index - len(matrix[0]) if y > 0 else index
    if dir == RIGHT:
        return index+1 if x < len(matrix[0]) - 1 else index
    if dir == DOWN:
        return index + len(matrix[0]) if y < len(matrix) - 1 else index
    if dir == LEFT:
        return index-1 if x > 0 else index


def diffJMatrix(matrix1, matrix2, display=True):
    """
    Compute the absolute mean error on 2 matrix

    Parameters
    ----------
    matrix1, matrix2 : 2D matrices
        Matrices from where one wants to compute the abs-mean error
    display : bool, optional
        Bool to display or not the result

    Returns
    -------
    matrix
        Matrix of same dimensions as matrix1 (and matrix2) containing the mean
        error for each cell
    float
        Absolute mean error for all cells
    """
    error_matrix = []
    mean = 0
    if display:
        print("Displaying ||J_est(x) - J(x)|| for all x:")

    for y in range(len(matrix1)):
        error_matrix.append([])
        for x in range(len(matrix1[0])):
            # Computing absolute mean error for the cell
            diff = abs(matrix1[y][x] - matrix2[y][x])
            mean += diff
            error_matrix[y].append(diff)
            # Display result
            if display:
                print(str(round(diff, 2)).rjust(8), end=' ')
        # Linebreaker
        if display:
            print(" ")
    # Computing absolute mean error for ALL cells
    cardinality = (y+1)*(x+1)
    mean /= cardinality
    return error_matrix, mean
