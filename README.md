# Snake AI using DQN

## Training result: 
<video src="https://user-images.githubusercontent.com/58980435/144391244-eef2bde7-11e8-4791-a976-98116f5a442f.mp4"></video>

## Set of inputs 
- direction of food + whether snake head will collide in each direction (faster convergence)

or
- nearest distance between snake head and collision objects + distance to food in 8 cardinal directions

## Summary of DQN
We have a brain (neural network) that has to make a decision given the inputs above. The issue at hand is how does the brain effectively decide which action is best?
This is where Q-learning comes in. In real life, we make our decisions based on how much value an action will bring us both in the present and in the future. For example, you would probably choose to eat healthier as its long-term health benefits outweighs the short-term satisfaction of eating unhealthy foods. Q-learning essentially replicates this idea. It is a formula to evaluate how good an action is by how much value it would bring in both the present and in the future.

A numerical value, Q-score, is used to evaluate how good an action is. The higher the score, the better the action. Given a state / inputs, the brain predicts a corresponding score for every action. The action with the highest score is then deemed to be the best action that should be taken.

How does the brain then learn over time?
In general, a neural network trains by adjusting its weights based on the difference between its output and target values. In order to create a set of target values for our brain to learn from, we require information from its past games. This is because in these past games we know every instance of the game's timeline and exactly how effective an action is in both the short-term and long-term. This would then allow us to calculate our target value or Q-score.

During the game, the brain remembers
- the previous state
- the action it took when in that state
- what was the reward it got from taking that action
- the current state it is now in

After every episode / instance of a game, the brain then uses this information to train itself. The target Q-score = (reward from the action + what it thinks is the future reward for the resulting state after the action * gamma). Gamma is some constant = 0.95 meant to slightly decrease the importance of future reward. This is because like in real life, the impact of a reward is also affected by when it is realised. For example, one would rather eat cake now than to eat it in the future because the joy from eating cake can be quickly realised.
