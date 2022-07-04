Snake AI using Q-learning

Originally wanted to use the entire snake board as input but didn't manage to make the model converge. Researched and found that very few people have managed to use this as input but still achieving roughly the same accuracy as a simplified / hackier input.

Inputs I tried using (both worked):

    distance to wall & snake body and distance to food in 8 cardinal directions
    whether the snake will collide the wall based on what move it does + direction of food (fastest convergence since this is the hackiest)

Training results: https://user-images.githubusercontent.com/58980435/144391244-eef2bde7-11e8-4791-a976-98116f5a442f.mp4

Will try to implement using NEAT algorithm.