import random
import numpy as np

class Game:
	def __init__(self, w, h, block_size):
		self.w = w
		self.h = h
		self.block_size = block_size
		self.pos = [(self.w//2, self.h//2)]
		self.vel = (1, 0)
		self.food = self.spawn_food()
		self.hunger = len(self.pos)*50

	def update(self, action):
		# move
		action -= 1 # [left, straight, right]
		vels = [(0, -1), (1, 0), (0, 1), (-1, 0)]
		dir = vels.index(self.vel)
		dir = (dir+action+4)%4
		self.vel = vels[dir]
		for i in range(len(self.pos)-1, 0, -1):
			self.pos[i] = self.pos[i-1]
		self.pos[0] = (self.pos[0][0]+self.vel[0], self.pos[0][1]+self.vel[1])

		done = self.game_over()
		if done:
			return -10, done

		self.hunger -= 1

		reward = 0
		if self.pos[0] == self.food:
			self.food = self.spawn_food()
			self.pos.append(self.pos[-1]) # grow
			self.hunger = len(self.pos)*50
			reward = 10
		# elif abs(self.pos[0][0]-self.food[0]) + abs(self.pos[0][1]-self.food[1]) <= 1:
		# 	reward = 0.2
		return reward, done

	def spawn_food(self):
		while 1:
			gen_pos = (random.randint(0, self.w-1), random.randint(0, self.h-1))
			ok = 1
			for pos in self.pos:
				if pos == gen_pos:
					ok = 0
					break 
			if ok:
				return gen_pos
	
	def game_over(self):
		# check collision
		if (not (0<=self.pos[0][0]<self.w)) or (not (0<=self.pos[0][1]<self.h)):
			return True
		for i in range(1, len(self.pos)):
			if self.pos[i] == self.pos[0]:
				return True
		# check hunger
		return self.hunger == 0
	
	def get_grid(self):
		grid = [[0 for j in range(self.w)] for i in range(self.h)]
		for x, y in self.pos:
			grid[x][y] = 1
		grid[self.food[0]][self.food[1]] = 3
		grid[self.pos[0][0]][self.pos[0][1]] = 2
		return grid
	
	def get_state(self):
		vels = [(0, -1), (1, 0), (0, 1), (-1, 0)]
		dir = vels.index(self.vel)
		one_hot_vel_dir = [0 for i in range(4)]
		one_hot_vel_dir[dir] = 1

		#return self.get_grid()

		hx, hy = self.pos[0]
		top, bot, left, right = hy, self.h-hy, hx, self.w-hx
		dtr, dbr, dbl, dtl = min(top, right), min(bot, right), min(bot, left), min(top, left)

		#temp = [top, dtr, right, dbr, bot, dbl, left, dtl]
		#temp2 = [temp[(i+1)%len(temp)] for i in range(len(temp))] # treat snake going top as top
		#state.extend(temp)

		for i in range(1, len(self.pos)):
			x, y = self.pos[i]
			dx, dy = x-hx, y-hy
			if dy > 0:
				bot = min(bot, dy)
			if dy < 0:
				top = min(top, -dy)
			if dx < 0:
				left = min(left, -dx)
			if dx > 0:
				right = min(right, dx)
			if abs(dx)==abs(dy):
				if dy < 0 and dx > 0:
					dtr = min(dtr, abs(dx))
				if dy > 0 and dx > 0:
					dbr = min(dbr, abs(dx))
				if dy > 0 and dx < 0:
					dbl = min(dbl, abs(dx))
				if dy < 0 and dx < 0:
					dtl = min(dtl, abs(dx))
		temp = [top, dtr, right, dbr, bot, dbl, left, dtl]
		#temp2 = [temp[(i+dir)%len(temp)] for i in range(len(temp))] # treat snake going top as top
		#state.extend(temp)
		#state = temp.copy()
		u, r, d, l = [i == dir for i in range(4)]
		state = []
		state.extend([
			(u and self.is_collision(vels[0])) or
			(r and self.is_collision(vels[1])) or
			(d and self.is_collision(vels[2])) or
			(l and self.is_collision(vels[3])),
			
			#left
			(r and self.is_collision(vels[0])) or
			(d and self.is_collision(vels[1])) or
			(l and self.is_collision(vels[2])) or
			(u and self.is_collision(vels[3])),

			#right
			(l and self.is_collision(vels[0])) or
			(u and self.is_collision(vels[1])) or
			(r and self.is_collision(vels[2])) or
			(d and self.is_collision(vels[3])),
		])

		state.extend(one_hot_vel_dir)

		fx, fy = self.food
		dfx, dfy = fx-hx, fy-hy
		f_dir = [dfy<0, dfx>0, dfy>0, dfx<0]
		#rotated_f_dir = [f_dir[(i+1)%len(f_dir)] for i in range(len(f_dir))]
		state.extend(f_dir)
		return state
	
	def is_collision(self, vel):
		pot = (self.pos[0][0]+vel[0], self.pos[0][1]+vel[1])
		if (not (0<=pot[0]<self.w)) or (not (0<=pot[1]<self.h)):
			return True
		for i in range(1, len(self.pos)):
			if self.pos[i] == pot:
				return True
		return False