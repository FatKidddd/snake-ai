import neat
from game import Game
import sys
import numpy as np
import time
import pygame
import os
import matplotlib.pyplot as plt
import pprint
import time
import math


def display(game):
	global SCREEN
	SCREEN.fill(BLACK)
	events = pygame.event.get()
	for event in events:
		if event.type == pygame.QUIT:
			pygame.quit()
			sys.exit()

	for pos in game.pos:
		i, j = pos
		rect = pygame.Rect(i*game.block_size, j*game.block_size, game.block_size, game.block_size)
		pygame.draw.rect(SCREEN, WHITE, rect)

	rect = pygame.Rect(game.food[0]*game.block_size, game.food[1]*game.block_size, game.block_size, game.block_size)
	pygame.draw.rect(SCREEN, RED, rect)

	pygame.display.update()


width, height = 10, 10
block_size = 20

BLACK = (0, 0, 0)
WHITE = (200, 200, 200)
RED = (255, 0, 0)

SCREEN = pygame.display.set_mode((width*block_size, height*block_size))
pygame.init()

def eval_genomes(genomes, config):
  e = 0
  for genome_id, genome in genomes:
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    game = Game(width, height, block_size)
    genome.fitness = 0
    while 1:
      if e == 0:
        display(game)
      state = game.get_state()
      prediction = net.activate(state)
      action = np.argmax(prediction)
      reward, done = game.update(action)
      # if game.pos[0] in game.past_points:
      #   genome.fitness -= 0.25
      # hx, hy = game.pos[0]
      # fx, fy = game.food
      # dfx, dfy = fy-hy, fx-hx
      # dist = math.sqrt(dfx**2+dfy**2)
      # if dist < 2:
      #   genome.fitness += 0.2
      if done:
        genome.fitness += reward
        break
    e += 1

def run(config_file):
  config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)

  p = neat.Population(config)
  #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-264')
  #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-499')

  p.add_reporter(neat.StdOutReporter(True))
  stats = neat.StatisticsReporter()
  p.add_reporter(stats)
  p.add_reporter(neat.Checkpointer(100))

  winner = p.run(eval_genomes, 300)

  print('\nBest genome:\n{!s}'.format(winner))

  print('\nOutput:')
  winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
  for i in range(10):
    game = Game(width, height, block_size)
    while 1:
      display(game)
      state = game.get_state()
      prediction = winner_net.activate(state)
      action = np.argmax(prediction)
      done = game.update(action)
      if done:
        break

  # node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
  # visualize.draw_net(config, winner, True, node_names=node_names)
  # visualize.plot_stats(stats, ylog=False, view=True)
  # visualize.plot_species(stats, view=True)

  #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
  #p.run(eval_genomes, 10)


if __name__ == '__main__':
  local_dir = os.path.dirname(__file__)
  config_path = os.path.join(local_dir, 'config')
  run(config_path)

plt.ion()

def plot(scores, avg_scores, losses):
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
