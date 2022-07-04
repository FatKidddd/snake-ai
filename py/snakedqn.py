from collections import deque
from dqn import DQNAgent
from game import Game
import sys
import numpy as np
import time
import pygame
import os
import matplotlib.pyplot as plt
import pprint
#from IPython import display

def display(game):
	global SCREEN
	SCREEN.fill(BLACK)
	events = pygame.event.get()
	for event in events:
		if event.type == pygame.QUIT:
			pygame.quit()
			sys.exit()

	# for i in range(game.w):
	# 	for j in range(game.h):
	# 		rect = pygame.Rect(i*game.block_size, j*game.block_size, game.block_size, game.block_size)
	# 		pygame.draw.rect(SCREEN, WHITE, rect, 1)

	for pos in game.pos:
		i, j = pos
		rect = pygame.Rect(i*game.block_size, j*game.block_size, game.block_size, game.block_size)
		pygame.draw.rect(SCREEN, WHITE, rect)

	rect = pygame.Rect(game.food[0]*game.block_size, game.food[1]*game.block_size, game.block_size, game.block_size)
	pygame.draw.rect(SCREEN, RED, rect)

	pygame.display.update()

plt.ion()

def plot(scores, avg_scores, losses):
	#display.clear_output(wait=True)
	#display.display(plt.gcf())
	plt.clf()
	plt.title('Training...')
	plt.xlabel('Number of Games')
	plt.ylabel('')
	plt.plot(scores, label='Score')
	plt.plot(avg_scores, label='Average Score')
	plt.plot(losses, label='Loss')
	plt.legend()
	plt.ylim(ymin=0)
	plt.text(len(scores)-1, scores[-1], str(scores[-1]))
	plt.text(len(avg_scores)-1, avg_scores[-1], str(avg_scores[-1]))
	plt.text(len(losses)-1, losses[-1], str(losses[-1]))
	plt.show(block=False)
	plt.pause(.1)

width, height = 20, 20
block_size = 20

BLACK = (0, 0, 0)
WHITE = (200, 200, 200)
RED = (255, 0, 0)

SCREEN = pygame.display.set_mode((width*block_size, height*block_size))
pygame.init()

#state_size = 229
state_size = 11
dqn = DQNAgent(state_size, 3)
#dqn = DQNAgent(height, width, 3)
dqn.load("weights4.h5")
episodes = 200
batch_size = 1000

x = [i+1 for i in range(episodes)]
scores, avg_scores, losses = [], [], []
total_score = 0

pp = pprint.PrettyPrinter()

e = 0
while 1:
	game = Game(width, height, block_size)
	e += 1

	# last_frames = deque([game.get_state()]*3)

	# while 1:
	# 	display(game)
	# 	state = np.array([last_frames])
	# 	action = dqn.get_action(state)
	# 	reward, done = game.update(np.argmax(action[0]))
	# 	reward = np.expand_dims(reward, axis=0)
	# 	done = np.expand_dims(done, axis=0)

	# 	next_frame = np.zeros((height, width))
	# 	if not done:
	# 		next_frame = game.get_state()

	# 	last_frames.append(next_frame)
	# 	last_frames.popleft()
	# 	next_state = np.array([last_frames])

	while 1:
		display(game)
		state = np.array([game.get_state()])
		action = dqn.get_action(state)
		reward, done = game.update(np.argmax(action[0]))
		reward = np.expand_dims(reward, axis=0)
		done = np.expand_dims(done, axis=0)
		next_state = np.array([game.get_state()])

		dqn.remember(state, action, reward, next_state, done)

		dqn.train_short_memory(state, action, reward, next_state, done)
		# if e > 300: 
		# 	time.sleep(0.1)
		if done:
			loss = dqn.train_long_memory(batch_size)
			losses.append(loss)
			scores.append(len(game.pos))
			total_score += len(game.pos)
			avg_scores.append(total_score/e)
			#plot(scores, avg_scores, losses)
			print("Episode {} ended with length {}, loss {}, epsilon {}".format(e+1, len(game.pos), loss, dqn.epsilon))
			break
	dqn.save("weights4.h5")