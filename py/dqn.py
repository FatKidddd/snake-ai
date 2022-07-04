from keras import Model
from keras.optimizers import SGD, Adam
from keras.layers import Input, Conv2D, Dense, Flatten
import tensorflow as tf
import numpy as np
from collections import deque
import random

class DQNAgent:
	def __init__(self, state_size, action_size):
		# self.rows = rows
		# self.cols = cols
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen=100000)
		self.gamma = 0.95 #discount rate
		self.epsilon = 1.0 #exploration rate
		self.epsilon_min = 0.001
		self.epsilon_decay = 0.995
		self.learning_rate = 0.001
		self.model = self.create_model()

	def create_model(self):
		inputs = Input(shape=(self.state_size,))
		hidden = Dense(128, activation='relu')(inputs)
		outputs = Dense(self.action_size, activation='softmax')(hidden)
		model = Model(inputs=inputs, outputs=outputs)

		# inputs = Input(shape=(3, self.rows, self.cols))
		# conv1 = Conv2D(16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', data_format='channels_first')(inputs)
		# #conv2 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', data_format='channels_first')(conv1)
		# flat = Flatten()(conv1)
		# hidden = Dense(128, activation='relu')(flat)
		# outputs = Dense(self.action_size, activation='softmax')(hidden)
		# model = Model(inputs=inputs, outputs=outputs)

		model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
		model.summary()
		return model

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def get_action(self, state):
		if np.random.rand() <= self.epsilon:
			if self.epsilon > self.epsilon_min:
				self.epsilon *= self.epsilon_decay
			idx = random.randrange(self.action_size)
			act_values = [0 for i in range(self.action_size)]
			act_values[idx] = 1
			return [act_values]
		return self.model.predict(state)

	def train_step(self, states, actions, rewards, next_states, dones):
		targets = self.model.predict(states)

		for i in range(len(dones)):
			q_new = rewards[i]
			if not dones[i]:
				next_state = np.expand_dims(next_states[i], axis=0)
				q_new = (rewards[i] + self.gamma * np.amax(self.model.predict(next_state)[0]))
			targets[i][np.argmax(actions[i])] = q_new

		history = self.model.fit(states, targets, epochs=1, verbose=0)
		loss = history.history['loss'][0]
		return loss
	
	def train_short_memory(self, state, action, reward, next_state, done):
		return self.train_step(state, action, reward, next_state, done)

	def train_long_memory(self, batch_size):
		minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
		states, actions, rewards, next_states, dones = zip(*minibatch)
		states = np.squeeze(states)
		actions = np.squeeze(actions)
		rewards = np.squeeze(rewards)
		next_states = np.squeeze(next_states)
		dones = np.squeeze(dones)
		return self.train_step(states, actions, rewards, next_states, dones)

	def load(self, name):
		self.model.load_weights(name)
	
	def save(self, name):
		self.model.save_weights(name)