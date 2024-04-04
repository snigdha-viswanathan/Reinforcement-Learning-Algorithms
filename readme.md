**Reinforcement Learning Algorithms**

| Algorithms | Environments Tested On |
| :---:   | :---: |
| Episodic Semi-Gradient n-Step SARSA | Mountain Car, Acrobot  |
| Reinforce with Baseline | Mountain Car, Acrobot  |
| One Step Actor Critic | Acrobot, Cartpole  |
| Prioritized Sweeping | Mountain Car, Frozen Lake, 687 Grid World  |

The learning curves are plotted for each of the combinations tested.

1. To run n step semi gradient sarsa,
     - a. Mountain Car environment
	     <br /> &emsp; python3 semi_gradient_sarsa/MountainCarSarsa.py
	     <br /> &emsp; python3 semi_gradient_sarsa/plotGraph.py
     - b. Acrobot environment
             <br /> &emsp; python3 semi_gradient_sarsa/AcrobotSarsa.py
             <br /> &emsp; python3 semi_gradient_sarsa/plotGraph.py
2. To run Reinforce with Baseline,
     - a. Mountain Car environment
	     <br /> &emsp; python3 reinforce_baseline/reinforce_baseline_mountaincar.py
     - b. Acrobot environment
             <br /> &emsp; python3 reinforce_baseline/reinforce_baseline_acrobot.py
3. To run One Step Actor Critic,
     - a. Acrobot environment
	     <br /> &emsp; python3 one_step_actor_critic/actor_critic_acrobot.py
     - b. CartPole environment
             <br /> &emsp; python3 one_step_actor_critic/actor_critic_cartpole.py

4. To run Prioritized Sweeping,
     - a. Mountain Car environment
	     <br /> &emsp; python3 prioritized_sweeping/prioritized_sweeping_mountaincar.py
     - b. Frozen Lake environment
             <br /> &emsp; python3 prioritized_sweeping/prioritized_sweeping_frozenlake.py
     - c. 687 Gridworld environment
             <br /> &emsp; python3 prioritized_sweeping/prioritized_sweeping_687gridworld.py

**Observations:**

- Episodic Semi-Gradient n-Step SARSA algorithm gave really good learning curves that converged and stabilized to a low number of steps on both Mountain Car and Acrobot environments.
- Reinforce with Baseline learnt really well on the Mountain Car environment and gave a smooth learning curve that converged and stabilized well. It converged to a good value in the Acrobot environment as well, but had some variance which can be reduced to observe better learning.
- One Step Actor Critic gave a good learning curve for the Acrobot environment that converges pretty well. However, on the CartPole environment it displays a lot of variance even though it reached the maximum reward possible.
- Prioritized sweeping requires an accurate model of the environment. Therefore, the algorithm displayed a good learning curve only after accurate tile coding for continuous state environments (Mountain Car). It displayed a relatively good learning curve on discrete state environments like 687 Gridworld and Frozen Lake.
