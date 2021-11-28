<!-- <<<<<<< HEAD -->
# CS698R-Project-Restless-Bandits-4
<!-- ======= -->
# Restless Multi Arm Bandit
## Abstract

The Restless Multi-Armed Bandit Problem (RMABP) is essentially a game between a player and an environment.
There are K arms and the state of each arm keeps evolving according to an underlying distribution and it transitions
at every time slot during the episode (one full play of the game). At each time slot, the player pulls one of the arms
and receives a reward. The goal of the game is to maximize the reward received over T time steps. DQN and PPO have been used with 
the memory matrix as the input state space which includes the last 'h' rewards of each state for further decision making. The 
updated approach gives more reward compared to SOTA input states.


## Research Survey 

Most existing work on RMABP solves the offline setting where the underlying working of the game is known.

In paper [[1]](#1) and [[2]](#2), authors build their work upon the Whittle Index Policy
which requires the knowledge of the underlying MDP which is rarely known in a practical setting. The main idea is
to assign an index to every arm and then choose the arm with the largest index and then update all indices and so
on. Parallel Q-learning recursions are utilized to learn the whittle indices which can then solve sub-problems with
reduced state space in order to maximize the average reward of the whole system in the long run. Whittle index
policy is asymptotically optimal under certain conditions given in [[3]](#3).

In paper [[4]](#4), authors solves the challenging generalization of the RMABP by developing a minimax regret objective i.e., minimizing the worst possible regret for the optimal policy and Proximal Policy
Approximation (PPO). It also takes into account any cost involved with taking actions. They also extend this
approach to a multi-agent setting.

## Problem Statement [[5]](#5)

<!-- ![](https://github.com/ArpitJIndal29/CS698R/blob/main/images/RMAB.JPG)  -->
![alt text](https://github.com/Exploration-Lab/CS698R-Project-Restless-Bandits-4/blob/main/images/RMAB.JPG)

The four-armed bandit being modelled has one non-terminal (starting) state and four
terminal states. The agent chooses one action vis-a-vis one arm. And this action places the
agent in one state, specific to that action with no stochasticity. The environment gives rewards, specific to an action. This reward is sampled from a Gaussian distribution of time-varying mean and a fixed variance, which makes the restless bandits different from a stationary one. This fixed variance which is common to all bandits can be seen as expected variance.
  
<div align="center" >
<b> R<sub>j</sub>(t) = &mu;<sub>j</sub>(t) + &epsilon;<sub>j</sub>(t) </b>
<br>
<b> &mu;<sub>j</sub>(t) = &lambda;&mu;<sub>j</sub>(t-1) + &kappa;<sub>j</sub> + &zeta;<sub>j</sub> </b>
</div>
<br />
where, &epsilon;<sub>j</sub> - &Nu;(0,&sigma;<sub>&epsilon;</sub>) and &zeta; - &Nu;(0,&sigma;<sub>&zeta;</sub>)
<br />
<br />

From the different parameter values of &sigma; and &kappa;, there can arise four cases;
1. Stable Variance with the trend,
2. Stable variance without trend, 
3. Variable variance with trend and 
4. Variable variance without trend. 

Unexpected variance is low and constant for stable variable cases whereas it is low for some periods and
high for some periods in variable variance cases. Trend value is zero for without trend cases.

## Proposed Solution

At first, we have used classical Reinforcement Learning techniques to train the agent. We have tried pure exploration
and pure exploitation, epsilon greedy, decaying epsilon greedy, softmax and UCB. Though UCB and Softmax worked
better than other algorithms, all these techniques got stuck at some sub-optimal level.

<!-- ![](https://github.com/ArpitJIndal29/CS698R/blob/main/images/reward.png) -->
![alt text](https://github.com/Exploration-Lab/CS698R-Project-Restless-Bandits-4/blob/main/images/reward.png)

We then moved to deep learning-based algorithms to get better results. We defined the whole game of 200 steps as an episode and each draw of arms as a time-step. And at each step, we defined the state of the environment as a collection of previous m actions and corresponding observed rewards, i.e. s<sub>t</sub> = [a<sub>t−1</sub>, r<sub>t−1</sub>, . . . , a<sub>t−m</sub>, r<sub>t−m</sub>]. We then trained the agent using DQN while considering m as a hyperparameter. This approach gave a significant improvement over classical algorithms but there is some flaw in the assumption that information about previous m actions and rewards is enough to predict the next step because if all the ‘m’ steps are of the same action the agent will have no information about other arm rewards. So, we modified our state as a 4*k matrix where i<sup>th</sup> row contains the last ‘k’ reward history of the
i<sup>th</sup> arm. After tuning hyperparameters, this agent takes optimal action almost
every time. We have then used DDQN and PPO to further improve the performance. 

## Experiments [[6]](#6) [[7]](#7)


### Hyperparameters for environment 
| Hyperparameter  | Value |
| ------------- | ------------- |
| k  | 8  |
| &gamma;  | 0  |
| &lambda;  | 0.99  |
| Expected Variance  | 2,4(low),16(high)  |
| &kappa;<sub>j</sub> | [-0.5,-0.5,0.5,0.5] |

### Design Choices for DQN
|   |  |
| ------------- | ------------- |
| Approximate action-value function  | Q(s, a; &theta;) |
| State-in values-out feedforward neural network  | Nodes : (32,400,300,4) |
| Loss Function  | L(&theta;<sub>i</sub>) = E<sub>s,a,r,s'</sub>[(r + &gamma; * max<sub>a'</sub>Q(s'; a'; &theta;<sub>i</sub>) - Q(s; a; &theta;<sub>i</sub>))<sup>2</sup>] |
| Decaying Epsilon-greedy Strategy  | &epsilon; decaying from 1 to 0.01 |
| learning rate | 0.01 |
| buffer size | 1000000 |
| batch size | 32 | 

### Design Choices for PPO
|   |  |
| ------------- | ------------- |
| Policy Network  | Nodes : (32,400,300,4) |
| Value Network  | Nodes : (32,400,300,4) |
| Loss Function  | L(&theta;,&theta;<sup>-</sup>) = E<sub>s,a,r,s' ~ &mu;(&theta; <sup>-</sup>)</sub>[max(G - V(s; &theta;), G - (V - clamp(V(s; &theta;) - V, -&delta;, &delta;)))] |
| learning rate | 0.01 |
| buffer size | 1000000 |
| batch size | 32 | 

## Results and Analysis

<div float:"left">
<img src="https://github.com/Exploration-Lab/CS698R-Project-Restless-Bandits-4/blob/main/images/dqn_training_trend_high.jpg" width="300">
<img src="https://github.com/Exploration-Lab/CS698R-Project-Restless-Bandits-4/blob/main/images/ppo_training_trend_high.jpg" width="300">
<img src="https://github.com/Exploration-Lab/CS698R-Project-Restless-Bandits-4/blob/main/images/dqn_ppo_evaluation_trend_high.jpg" width="300">
</div>

The reward function of the environment is noisy and the agent does not have access to environment dynamics and
hence it can never reach the true achievable reward due to the randomness of the reward itself.
We plot the training performance of the DQN agent by comparing the evolution of reward with the
true achievable reward. For the first 100 episodes the epsilon is decayed linearly from 1 to 0.01, and we
see that as soon as epsilon settles, the DQN performance peaks as it starts exploiting the learnt policy and then
maintains its performance.
Next, we do the same for the PPO agent. PPO uses action space noise and is able to climb up to the
peak performance much before 100 episodes without any falls. This is because PPO uses clipping while updating
the gradients and therefore abrupt changes are avoided and hence it gradually moves towards the optimal policy
instead of fluctuating to bad policies.
Lastly, we evaluate our trained DQN and PPO agents on 100 instances of the game (environment) and
average out the results. We also contrast the performance of our trained agents with true achievable reward and
a random scheme in which an action is chosen at random at every time step. We observe that PPO outperforms
DQN, especially in the later part of the game when trend and high volatility takes full effect.

<div float:"left">
<img src="https://github.com/Exploration-Lab/CS698R-Project-Restless-Bandits-4/blob/main/images/1game_dqn_evaluation_trend_high.jpg" width="450">
<img src="https://github.com/Exploration-Lab/CS698R-Project-Restless-Bandits-4/blob/main/images/1game_ppo_evaluation_trend_high.jpg" width="450">
</div>

We observe that as soon as the value of the arm that the agent is following falls down, agent tries to switch the arm and is successful in most cases.

##  Future Directions and Conclusions
It can be observed that, because the environment is highly stochastic, the agent can never achieve the optimal
rewards. Also, the agent not only learns on the basis of the rewards it may achieve having selected an action, but
also from the memory matrix. The memory matrix allows the agent to learn the general distribution of all the
arms and exactly when to explore instead of exploiting the arm with high reward. Currently, we focussed on a
single agent setting where a single arm was selected out of the four bandits. So, the optimal action only depended
on the reward it achieved after selecting the action. This work can be extended to a multi-agent setting where
multiple scenarios can be explored, for example, the arm selection of all the agents are trained on the basis of their
cumulative reward. Similarly, there can be an extended condition where all the agents cannot select the same arm
and hence they will be inter dependent.
Here, in our problem statement, the total time steps for which the agent has to choose an action is already
known. If there in no finite time steps given beforehand, then the agent has to be trained in a online setting. The
agent need to be trained in a very robust environment with a large dataset so that the neural network can ganeralize
each and every situation that can arise in any time step.


## References
<a id="1">[1]</a>
Jing Fu, Yoni Nazarathy, Sarat Moka, and Peter G Taylor. Towards q-learning the whittle index for restless bandits.
In 2019 Australian & New Zealand Control Conference (ANZCC), pages 249–254. IEEE, 2019.

<a id="2">[2]</a>
Vivek S Borkar and Karan Chadha. A reinforcement learning algorithm for restless bandits. In 2018 Indian Control
Conference (ICC), pages 89–94. IEEE, 2018.

<a id="3">[3]</a>
Richard R Weber and Gideon Weiss. On an index policy for restless bandits. Journal of applied probability, 27(3):
637–648, 1990.

<a id="4">[4]</a>
Jackson A Killian, Lily Xu, Arpita Biswas, and Milind Tambe. Robust restless bandits: Tackling interval uncertainty
with deep reinforcement learning. arXiv preprint arXiv:2107.01689, 2021.

<a id="5">[5]</a>
Konstantinidis E Speekenbrink M. Uncertainty and exploration in a restless bandit problem. pages 351–67, 2015.
doi: 10.1111/tops.12145.

<a id="6">[6]</a>
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin
Riedmiller. Playing atari with deep reinforcement learning, 2013.

<a id="7">[7]</a>
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization
algorithms, 2017.
<!-- >>>>>>> CS698R/main -->
