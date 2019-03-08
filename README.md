# INFO8003-1 - Optimal decision making for complex problems
## Assignment 1 - Section 1 to 7
This file comments on how to use the code done in the first assignment. In order to do so, one should run the following :
```sh
$ python3 run.py <options>
```
When called without option, the code runs the question 6 using 100 episodes of 1000 transitions without any experience replay with alpha = 0.05 and epsilon = 0.5 in deterministic mode. In order to change that, one can use the following parameters:
* `--Q` **{2, 3, 4, 5}** : Select the question to run
* `--N`  **int** : Number of iterations (>0).
* `--t`  **int** : Size of the trajectory (>0). Allow to tune history during Q5+
* `--beta` **float**: Value of the Beta parameter. Should be between 0 and 1.
* `--epsilon` **float**: Value of the Epsilon parameter for the e-greedy policy in Q6. Should be between 0 and 1.
* `--learning_ratio` **float**: Learning ratio used during the Q-learing algorithm. Should be between 0 and 1.
* `--nb_episodes` **int** : Number of episodes used during the Q-learing algorithm (>0).
* `--batch_size` **int** : Batch size for the experience replay use in Q6. 
* `--discount_factor` **float** : Discount factor (gamma). Should be change to see the impact of this parameter in Q7.
