from grid import *
from batch import *
from copy import copy, deepcopy
import math

###############################################################################
#                               MDP Class                                    #
###############################################################################


class MDP:
  def __init__(self, beta, discount_factor=0.99, learning_ratio=0.0, b_s=5000):
    """
    Parameters
    ----------
    beta : float
      Parameter for stochastic setting.
    discount_factor : float, optional
      Parameter for the discount factor. Corresponds to gamma in theory
    learning_ratio : float, optional
      Parameter for the learning ratio. Corresponds to alpha in theory
    b_s : int, optional
      Length of the batch used in the Q6
    """
    self.__grid = Grid()
    self.beta = beta
    self.__gamma = discount_factor
    self.__alpha = learning_ratio
    self.batch = Batch(b_s)
    self.batch_size = b_s

  def computeRP(self):
    """
    Compute the transitions matrix and reward signal of the MDP.

    Returns
    -------
    list[list[int]]
      r(x, u)
    list[list[list[float]]]
      p(x'|x, u)
    """
    # Init
    r = []
    p = []
    # Working with indexes. (0,0) -> 0. (1,1) -> 6
    for i in range(len(self.__grid.matrix1D)):
      r.append([])
      p.append([])

      # Corresponding x and y
      x = i % self.__grid.width
      y = i // self.__grid.width

      for u in range(NUMBER_ACTIONS):
        p[i].append([])
        # Computing r(x, u) for all u in U, x in X
        r[i].append((1 - self.beta) * (
                  getFromMatrix(u, self.__grid.matrix, x, y)) + (self.beta) * (
                  getFromMatrix(UP, self.__grid.matrix, 0, 0)))

        # Getting the index of arrival after action u in x, ie , x'
        target_j = getIndexFromMatrix(u, self.__grid.matrix, i)
        for j in range(len(self.__grid.matrix1D)):
          # Computing p(x'|x, u)
          if j == 0:
            if target_j == 0:
              # (0, 0) and going to (0, 0)
              p[i][u].append(1)
            else:
              # Chance of going to (0, 0) while not taking the corresp. action
              p[i][u].append(self.beta)
          elif j == target_j:
            # Chance of reaching to x'
            p[i][u].append(1 - self.beta)
          else:
            p[i][u].append(0)
    return r, p

  def __computeQ(self, N=5, history=None):
    """
    Compute Q(x, u) from r(x, u) and p(x'|x, u).
    This function works both when estimating r and p (history != None) and
    while workong with the real values of r and p (history = None)

    Parameters
    ----------
    N : int, optional
      Number of iterations.
    history : list, optional
      Trajectory to estimate r(x, u) and p(x'|x, u)

    Returns
    -------
    list[list[float]]
      Q(x, u)
    """
    # Getting r and p
    # Real values
    if history is None:
      r, p = self.computeRP()
    # Estimated values
    else:
      r, p = self.computeRPfromHistory(history)
    # Iterative methods : initialization of the previous Q to 0 as Q_0 = 0 for
    # all x in X, u in u as defined in theory
    previous_Q = [[0, 0, 0, 0]for j in range(len(self.__grid.matrix1D))]
    # Creating Q of same size
    Q = deepcopy(previous_Q)

    # N iterations
    for n in range(N):
      # for all x in X
      for i in range(len(self.__grid.matrix1D)):
        # for all u in U
        for u in range(NUMBER_ACTIONS):
          sum_p = 0
          # Computing second term
          for j in range(len(self.__grid.matrix1D)):
            sum_p += p[i][u][j] * self.__maxQ(previous_Q, j)
          # Computing Q
          Q[i][u] = r[i][u] + (self.__gamma * sum_p)
      # Updating previous Q
      previous_Q = deepcopy(Q)
    return Q

  def getPolicyFromQ(self, Q):
    """
    Compute the optimal policy from the computation of Q(x, u).
    This function works both when estimating r and p (history != None) and
    while workong with the real values of r and p (history = None)

    Parameters
    ----------
    Q : list[list[float]]
      Q(x, u)

    Returns
    -------
    list[{0, 1, 2, 3}]
      mu^*(x) : the optimal stationary policy.
    """
    policy = []

    for x in range(len(self.__grid.matrix1D)):
      # Getting the best action from Q for all x in X
      policy.append(self.__bestAction(Q, x))
    return policy

  def getQ(self, N=1000, history=None):
    """
    Compute the optimal policy from the computation of Q(x, u).
    This function works both when estimating r and p (history != None) and
    while workong with the real values of r and p (history = None)

    Parameters
    ----------
    N : int, optional
      Number of iterations.
    history : list[x0, u0, r0, x1, ... xt−1, ut−1, rt−1, xt], optional
      Trajectory of size T

    Returns
    -------
    list[list[float]]
      Q(x, u)
    """
    # Real values
    if history is None:
      Q = self.__computeQ(N)
    # Estimated values
    else:
      Q = self.__computeQ(N, history)

    return Q

  def __bestAction(self, matrix, index):
    """
    Compute the optimal action to take based on Q and the position in the grid

    Parameters
    ----------
    Q : list[list[float]]
      Q(x, u)
    index : int
      Position in the grid using indexes

    Returns
    -------
    {0, 1, 2, 3}
      The optimal direction to take from index in the grid based on Q(x, u)
    """
    action = -1
    # initialization of max
    max = -float('Inf')
    # Simple max algo for all actions u in U
    for i in range(NUMBER_ACTIONS):
      if matrix[index][i] > max:
        max = matrix[index][i]
        action = i

    return action

  def __maxQ(self, Q, index):
    """
    Compute the best Q(x, u) for all x in X

    Parameters
    ----------
    Q : list[list[float]]
      Q(x, u)
    index : int
      Position in the grid using indexes

    Returns
    -------
    float
      max_{u in U} Q(index, u) : max Q in all the direction possible
    """
    max = -float('Inf')
    for i in range(NUMBER_ACTIONS):
      max = Q[index][i] if Q[index][i] > max else max
    return max

  def createHistory(self, T=50, policy=None, starting_pos=None, epsilon=0.0):
      """
      Create a feasible trajectory of size T. Can be totally random or follow
      a specific policy. Can start at random or at a starting position

      Parameters
      ----------
      T: int, optional
        Size of the trajectory
      policy : list[{0, 1, 2, 3}], optional
        The policy to follow if wanted
      starting_pos : int
        The index of the starting position
      epsilon : float; optional
        Epsilon parameter for the e-greedy policy

      Returns
      -------
      list[x0, u0, r0, x1, ... xt−1, ut−1, rt−1, xt]
        Trajectory of size T
      """
      h = []
      if starting_pos is None:
        # Getting a random starting position
        starting_pos = random.randint(0, len(self.__grid.matrix1D)-1)
      for t in range(0, T):
        # Initial case : start at starting position
        if t == 0:
          next_x = starting_pos

        h.append(next_x)

        if policy is None:
          # Random legal action
          action = random.randint(0, NUMBER_ACTIONS-1)
        else:
          r = random.uniform(0, 1)
          if r < epsilon:
            # Getting a random action
            action = random.randint(0, NUMBER_ACTIONS-1)
          else:
            action = policy[next_x]
        h.append(action)

        # Computing the next x
        if random.random() <= self.beta:
          next_x = 0
        else:
          next_x = getIndexFromMatrix(action, self.__grid.matrix, next_x)
        # Adding the reward for going to the next x
        h.append(self.__grid.getRewardIndex(next_x))
      # Adding xt, the last position
      h.append(next_x)
      return h

  def computeRPfromHistory(self, history):
    """
    Compute the estimated transitions matrix and the estimated reward signal
    of the MDP based on a trajectory

    Parameters
    ----------
    history : list, optional
      Trajectory to estimate r(x, u) and p(x'|x, u)

    Returns
    -------
    list[list[int]]
      ^r(x, u)
    list[list[list[float]]]
      ^p(x'|x, u)
    """
    # cardinality will be of size r.
    cardinality = []
    r = []
    p = []
    # Initialization of r, p and cardinality
    for i in range(len(self.__grid.matrix1D)):
      r.append([])
      p.append([])
      cardinality.append([])
      for u in range(NUMBER_ACTIONS):
        p[i].append([])
        # Initial reward for unkown cases = 0. See report
        r[i].append(0)
        cardinality[i].append(0)
        for j in range(len(self.__grid.matrix1D)):
          # Initial transition for unknown transitions = 0. See report
          p[i][u].append(0)

    # Searching in the history
    for t in range(0, len(history)-1, 3):
      # Increasing the cardinality, the reward signal and the transition matrix
      cardinality[history[t]][history[t+1]] += 1
      r[history[t]][history[t+1]] += history[t+2]
      p[history[t]][history[t+1]][history[t+3]] += 1

    # Divding the sum of rewards and transistion encountered by the cardinality
    for i in range(len(self.__grid.matrix1D)):
      for u in range(NUMBER_ACTIONS):
        if cardinality[i][u] != 0:
          r[i][u] /= cardinality[i][u]
          for j in range(len(self.__grid.matrix1D)):
            p[i][u][j] /= cardinality[i][u]
    return r, p

  def meanError(self, r_est, r, p_est, p):
    """
    Compute the absolute mean error between r and r_estimated and the root
    mean square error between p and p_estimated

    Parameters
    ----------
    r_est : list[list[int]]
      Estimated reward signal
    r : list[list[int]]
      Real reward signal
    p_est : list[list[[list[float]]]
      Estimated transitions matrix
    p : list[list[[list[float]]]
      Real transitions matrix
    Returns
    -------
    float
      Absolute mean error on the reward signal
    float
      Root mean square error on the transitions matrix
    """
    # Init
    mean_r = 0
    mean_p = 0
    cardinality_r = 0
    cardinality_p = 0

    for i in range(len(self.__grid.matrix1D)):
      for u in range(NUMBER_ACTIONS):
        # Absolute mean error
        mean_r += abs(r[i][u] - r_est[i][u])
        cardinality_r += 1
        for j in range(len(self.__grid.matrix1D)):
          # Root mean square error
          mean_p += math.pow(p[i][u][j] - p_est[i][u][j], 2)
          cardinality_p += 1
    mean_r /= cardinality_r
    mean_p /= cardinality_p
    mean_p = math.sqrt(mean_p)
    return mean_r, mean_p

  def getQlearning(self, nb_episodes, T, epsilon):
    """
    Estimate Q using the Q_learning algorithm.

    Parameters
    ----------
    nb_episodes : int
      Number of eposides used in the Q-learing algorithm
    T : int
      Size of the trajectories, ie the number of transitions.
    epsilon : float
      Epsilon parameter for the e-greedy policy

    Returns
    -------
    list[list[float]]
      Q(x, u)
    """
    # Initialization of Q variables
    Q = [[0, 0, 0, 0]for j in range(len(self.__grid.matrix1D))]
    policy = self.getPolicyFromQ(Q)
    # For the number of episodes
    for k in range(nb_episodes):
      # Getting a trajectory following the current Q computed
      s = 18
      for t in range(T):
        exp = self.createHistory(1, policy, s, epsilon)
        next_s = exp[3]
        if self.batch.isFull():
          for h in random.sample(self.batch.batch, self.batch_size):
            x = h[0]
            u = h[1]
            r = h[2]
            x_next = h[3]

            Q[x][u] = (1 - self.__alpha) * Q[x][u] + (self.__alpha) * (
                       r + self.__gamma * self.__maxQ(Q, x_next))
          self.batch.empty()
          policy = self.getPolicyFromQ(Q)

        self.batch.add(exp)
        s = next_s
      epsilon -= 0.0045

    return Q
  def getQlearningFromTrajectories(self, list_traj):
    """
    Estimate Q using the Q_learning algorithm.

    Parameters
    ----------
    list_traj : list[list[x0, u0, r0, x1, ... xt−1, ut−1, rt−1, xt]]
      `Number of eposides` trajectories of size T

    Returns
    -------
    list[list[float]]
      Q(x, u)
    """
    # Initialization of Q variables
    Q = [[0 , 0, 0, 0]for j in range(len(self.__grid.matrix1D))]

    for k in range(len(list_traj)):
      for t in range(0, len(list_traj[0])-1, 3):
        x = list_traj[k][t]
        u = list_traj[k][t+1]
        r = list_traj[k][t+2]
        x_next = list_traj[k][t+3]

        Q[x][u] = (1 - self.__alpha) * Q[x][u] + (self.__alpha) * (
                   r + self.__gamma * self.__maxQ(Q, x_next))

    return Q
