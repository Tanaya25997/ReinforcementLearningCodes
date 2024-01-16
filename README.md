# ReinforcementLearningCodes


1. RL 1: Random algorithm to to estimate J(π) by simulating many of the possible outcomes that might result from running π on an MDP (MDP defined in the main file).  J(π) is defined as the expected discounted
return. We can construct an estimate of J(π), J^(π), by averaging the discounted returns observed across N simulations.

2. RL 2: Implement the Evolution Strategy method on the Cartpole Domain.

The Evolution Strategy (ES) method for policy search is a simple BBO algorithm that has achieved remarkable
performance on domains like playing Atari games and controlling simulated MuJoCo robots. Here in the implementation of the
version of ES originally introduced in the work of Salimans et al. (2017) - https://arxiv.org/abs/1703.03864

The Cartpole domain  consists of a pole attached to a cart that can move horizontally along a track. The pole is placed upright 
on the cart, and the goal of the RL agent is to balance the pole by moving the cart to the left or right.

