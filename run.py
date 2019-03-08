import time
from plot import *
from agent import *
from MDP import *
from argparse import ArgumentParser, ArgumentTypeError

if __name__ == "__main__":
    random.seed(5)
    start = time.time()
    usage = """
    USAGE:      python3 run.py <options>
    EXAMPLES:   (1) python run.py
                    - Launch the Q6 of the project : computing Jn
    """

    # Using argparse to select the different setting for the run
    parser = ArgumentParser(usage)

    # beta : corresponds to the "Beta" from the report.
    parser.add_argument(
        '--beta',
        help="""Value of the Beta parameter while being in stochastic setting.
        Not filling this parameter will put the code in deterministic mode.
        """,
        type=float,
        default=0
    )

    # N : corresponds to the number of iterations
    parser.add_argument(
        '--N',
        help='Number of iterations',
        type=int,
        default=1000)

    # Q : Allow to select the question the user want to run.
    parser.add_argument(
        '--Q',
        help='Selection the question of the project one wants to run',
        type=int,
        default=6
    )

    # t: Size of the hisotry
    parser.add_argument(
        '--t',
        help='Size of the history one wants to consider',
        type=int,
        default=1000
    )

    # discount_factor : gamma parameter
    parser.add_argument(
        '--discount_factor',
        help='Discount factor (gamma)',
        type=float,
        default=0.99
    )

    # epsilon : e parameter for e-greedy policy
    parser.add_argument(
        '--epsilon',
        help='Epsilon parameter used in the e-greedy policy for Q-learning',
        type=float,
        default=0.5
    )

    # learning_ratio : alpha parameter for Q6
    parser.add_argument(
        '--learning_ratio',
        help='Learning ratio used during the Q-learing algorithm',
        type=float,
        default=0.05
    )

    # nb_episodes
    parser.add_argument(
        '--nb_episodes',
        help='Number of episodes used during the Q-learing algorithm',
        type=int,
        default=100
    )

    # batch size
    parser.add_argument(
        '--batch_size',
        help="""Batch size for the experience replay use in Q6.""",
        type=int,
        default=1000
    )
    # Parsing the arguments
    args = parser.parse_args()
    nb_iterations = args.N
    beta = args.beta
    question_number = args.Q
    t = args.t
    gamma = args.discount_factor
    epsilon = args.epsilon
    alpha = args.learning_ratio
    nb_episodes = args.nb_episodes
    batch_size = args.batch_size

    mode = "deterministic" if beta == 0 else "stochastic"

    if beta < 0 or beta > 1 or gamma < 0 or gamma > 1 or \
            epsilon < 0 or epsilon > 1 or alpha < 0 or alpha > 1:
        print("The alpha/beta/gamma/epsilon parameters should"
              " be between 0 and 1.")
        exit()
    if nb_iterations <= 0 or t <= 0 or nb_episodes <= 0:
        print("The number of iterations, the size of history and "
              "the number of episodes should be +.")
        exit()
    if batch_size <= 0:
        print("The batch size should be positive.")
        exit()
    print("Beta is equals to {}."
          " The program will therefore run in {} mode.".format(beta, mode))

    if question_number == 2:
        # Creating a new agent
        new_agent = Agent(beta, gamma)
        # Display the simple "Up Policy"
        new_agent.simpleUpPolicyDisplay(nb_iterations)
    elif question_number == 3:
        # Creating a new agent
        new_agent = Agent(beta, gamma)
        # Creating a simple policy of going down:
        policy = new_agent.getSimplePolicy(DOWN)
        print("Displaying J{} for the simple policy of"
              " always going down:".format(nb_iterations))
        # Display the Jn for the {nb_iterations}th first iterations
        new_agent.computeJ(policy, nb_iterations)
    elif question_number == 4:
        new_mdp = MDP(beta, gamma)
        # Computing Q
        real_Q = new_mdp.getQ(nb_iterations)
        # Getting optimal policy
        opt_policy = new_mdp.getPolicyFromQ(real_Q)
        print("Printing J{} using the"
              " optimal policy:".format(format(nb_iterations)))
        new_agent = Agent(beta, gamma)
        # Corresponding JN
        J_opt = new_agent.computeJ(opt_policy, nb_iterations)

    elif question_number == 5:
        new_mdp_est = MDP(beta, gamma)
        new_mdp = MDP(beta, gamma)
        # Creating random history
        history = new_mdp_est.createHistory(t)

        # Mean error on r_est and p_est
        r, p = new_mdp.computeRP()
        r_est, p_est = new_mdp_est.computeRPfromHistory(history)
        mean_r, mean_p = new_mdp.meanError(r_est, r, p_est, p)
        print("Mean absolute deviation"
              " on the reward signal : {}.".format(mean_r))
        print("Root-mean-square deviation on "
              "the transition probabilities : {}.".format(round(mean_p, 4)))

        # Computation of ^J_µ*^N
        # Getting Q_est
        Q_est = new_mdp_est.getQ(nb_iterations, history)
        # Optimal policy estimated
        opt_policy_est = new_mdp_est.getPolicyFromQ(Q_est)
        new_agent = Agent(beta, gamma)
        print("Printing J{} using the"
              " optimal policy estimated :".format(format(nb_iterations)))
        # Corresponding ^JN
        J_est = new_agent.computeJ(opt_policy_est, nb_iterations)

        # Computation of J_µ*^N
        # Getting real_q
        Q_real = new_mdp.getQ(nb_iterations)
        # Real optimal policy
        opt_policy = new_mdp.getPolicyFromQ(Q_real)
        print("Printing J{} using the"
              " optimal policy:".format(format(nb_iterations)))
        # Corresponding JN
        J = new_agent.computeJ(opt_policy, nb_iterations)

        # Display the difference between J_est and J
        J_error, mean_J = diffJMatrix(J, J_est)
        print("Absolute mean deviation on J(x) : {}".format(round(mean_J, 2)))

    elif question_number == 6:
        agent = Agent(beta, gamma)
        mdp = MDP(beta, gamma, alpha, batch_size)
        print("Part 0 : Printing cumultative reward using the real optimal"
              " policy.")
        # Computation of J_µ*^N
        # Getting real_q
        Q_real = mdp.getQ(nb_iterations)
        # Real optimal policy
        opt_policy = mdp.getPolicyFromQ(Q_real)
        print("Printing J{} using the"
              " optimal policy:".format(format(nb_iterations)))
        # Corresponding JN
        J = agent.computeJ(opt_policy, nb_iterations)

        print("Part 1 : Using {} random trajectories".format(nb_episodes))
        # nb_episodes should be 100 and t 1000
        # PART 1 : Qlearning from history
        list_trajectories = []
        # Creating the list of trajectories
        for k in range(nb_episodes):
            list_trajectories.append(mdp.createHistory(t))

        Q_est_history = mdp.getQlearningFromTrajectories(list_trajectories)
        # Computing the estimated optimal policy for the Q computed
        opt_policy_est_history = mdp.getPolicyFromQ(Q_est_history)

        # Corresponding ^JN
        J_est = agent.computeJ(opt_policy_est_history, nb_iterations)

        # Display the difference between J_est and J
        J_error, mean_J = diffJMatrix(J, J_est)
        print("Absolute mean deviation on J(x) : {}".format(round(mean_J, 2)))

        Q_error, mean_Q = diffJMatrix(Q_real, Q_est_history, display=False)
        print("Absolute mean deviation on "
              "Q(x, u) : {}".format(round(mean_Q, 2)))

        # PART 2 : Qlearning from policy
        print("----------------------------------")
        print("Part2: Using epsilon-greedy policy:")
        # Computing Q using q_learning algorithm
        Q_est = mdp.getQlearning(nb_episodes, t, epsilon)

        # Computing the estimated optimal policy for the Q computed
        opt_policy_est = mdp.getPolicyFromQ(Q_est)

        # Corresponding ^JN
        J_est = agent.computeJ(opt_policy_est, nb_iterations)

        # Display the difference between J_est and J
        J_error, mean_J = diffJMatrix(J, J_est)
        print("Absolute mean deviation on J(x) : {}".format(round(mean_J, 2)))

        Q_error, mean_Q = diffJMatrix(Q_real, Q_est, display=False)
        print("Absolute mean deviation on "
              "Q(x, u) : {}".format(round(mean_Q, 2)))

    print("--------- Comp. time : {} ---------".format(time.time() - start))
