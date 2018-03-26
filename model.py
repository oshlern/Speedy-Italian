from keras.models import *#Sequential
from keras.layers import *#Conv2D, Flatten, Dense
from keras.losses import *#mean_squared_error
from keras.optimizers import *#SGD
import random
from collections import deque
import csv

class QNet:
	# Define Q neural network
	def __init__(self, state_shape, action_size, hyperparams={'max_len': 2000, 'discount_rate': 0.95, 'exploration_init': 1.0, 'exploration_fin': 0.005, 'exploration_decay': 0.99, 'learning_rate': 0.000003, 'batch_size': 32}):
		self.state_size = 1
		for shp in state_shape:
			self.state_size *= shp
		self.input_shape = state_shape
		self.action_size = action_size
		self.memory = deque(maxlen=hyperparams['max_len'])
		self.batch_size = hyperparams['batch_size']
		self.discount_rate = hyperparams['discount_rate']
		self.exploration_init = hyperparams['exploration_init']
		self.exploration_decay = hyperparams['exploration_decay']
		self.exploration_fin = hyperparams['exploration_fin']
		self.exploration = self.exploration_init
		self.learning_rate = hyperparams['learning_rate']
		self.model = self._build_model()

	# Build a model
	def _build_model(self, layer_sizes=[20,20], conv_layer_sizes=[32,64,64]):
		model = Sequential()
		if len(self.input_shape) != 1:
			model.add(Conv2D(conv_layer_sizes[0], kernel_size=(8, 8), strides=(4, 4), activation='sigmoid', input_shape=self.input_shape))
			model.add(Conv2D(conv_layer_sizes[1], kernel_size=(4, 4), strides=(2, 2), activation='sigmoid'))
			model.add(Conv2D(conv_layer_sizes[2], kernel_size=(3, 3), strides=(1, 1), activation='relu'))
			model.add(Flatten())
			model.add(Dense(256, activation='relu'))
		else:
			model.add(Dense(layer_sizes[0], activation='sigmoid', input_shape=self.input_shape))
		for size in layer_sizes[1:]:
			model.add(Dense(size, activation='relu'))
		model.add(Dense(self.action_size, activation='linear'))
		model.compile(loss=mean_squared_error, # TODO: better loss, why categorical?
			optimizer=SGD(lr=self.learning_rate), # TODO: better optimizer, Adam
			metrics=['accuracy'])
		return model

	def process_state(self, state): # TODO: is this needed? can we not input an individual state? if so, can we just input [state]?
		return np.array([state]) #state.reshape(1, state.shape[0], state.shape[1], state.shape[2])

	def remember(self, state, actions, reward, next_state, done):# remember our information
		self.memory.append((state, actions, reward, next_state, done))# 	Just add it to memory
		
	def act(self, state):
		if self.exploration > np.random.rand():# if we are exploring
			return random.randrange(self.action_size) # random action
		else:
			outcomes = self.model.predict(self.process_state(state)) # choose the action in our current state that returns the highest value
			return np.argmax(outcomes[0])

	def test_act(self, state):
		outcomes = self.model.predict(self.process_state(state)) # choose the action in our current state that returns the highest value
		return np.argmax(outcomes[0])

	def updateTargetNet(self):
		self.target_model = clone_model(self.model, self.custom_model_objects)
        self.target_model.compile(optimizer='sgd', loss='mse')
	
	def train(self, action_num): # train our model
		# training_data = list(self.memory)[len(self.memory)-action_num: len(self.memory)] # take the most recent iterations of the model
		training_data = random.sample(list(self.memory), self.batch_size)
		states, actions, rewards, next_states, dones = zip(*training_data)
		states, actions, rewards, next_states, dones = np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
		terminal_states, terminal_actions, terminal_rewards = states[dones], actions[dones], rewards[dones]
		not_dones = np.invert(dones)
		nonterminal_states, nonterminal_actions, nonterminal_rewards, nonterminal_next_states = states[not_dones], actions[not_dones], rewards[not_dones], next_states[not_dones]
		predicted_action_vals = self.model.predict(nonterminal_next_states)
		nonterminal_rewards = nonterminal_rewards + self.discount_rate * np.amax(predicted_action_vals, axis=1) # set the Q function output as: current reward + discount factor * best future reward based on next state
		states = np.append(terminal_states, nonterminal_states, axis=0)
		rewards = np.append(terminal_rewards, nonterminal_rewards, axis=0)
		actions = np.append(terminal_actions, nonterminal_actions, axis=0)
		target_rewards = np.array(self.model.predict(states))
		reorder = np.random.permutation(self.batch_size)
		states, target_rewards, rewards, actions = states[reorder], target_rewards[reorder], rewards[reorder], actions[reorder]
		if random.random() < 0.006:
			# print([(rewards[i], target_rewards[i,actions[i]]) for i in range(self.batch_size)])
			print(rewards)
			print([target_rewards[i,actions[i]] for i in range(self.batch_size)])
			print("__________AVG MSE (diff between reward and predicted) = ", np.sqrt(sum([(rewards[i] - target_rewards[i,actions[i]])**2 for i in range(self.batch_size)])/self.batch_size))
		for i in range(self.batch_size):
			target_rewards[i,actions[i]] = rewards[i]
		self.model.fit(states, target_rewards, epochs=1, verbose=0) # fit our model based on our state and target value
		# decay exploration if possible
		if self.exploration > self.exploration_fin:
			self.exploration *= self.exploration_decay



