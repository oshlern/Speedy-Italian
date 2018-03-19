import keras

class QNet:
	# Define Q neural network
	def __init__(self, state_shape, action_size, hyperparams={'max_len': 2000, 'discount_rate': 0.99, 'exploration_init': 1.0, 'exploration_fin': 0.005, 'exploration_decay': 0.99, 'learning_rate': 0.3}):
		self.state_size = 1
		for shp in state_shape:
			self.state_size *= shp
		self.input_shape = state_shape
		self.action_size = action_size
		self.memory = deque(maxlen=hyperparams['max_len'])
		self.discount_rate = hyperparams['discount_rate']
		self.exploration_init = hyperparams['exploration_init']
		self.exploration_decay = hyperparams['exploration_decay']
		self.exploration_fin = hyperparams['exploration_fin']
		self.learning_rate = hyperparams['learning_rate']
		self.model = self._build_model()

	# Build a model
	def _build_model(self):
		model = Sequential()
		model.add(Conv2D(32, kernel_size=(8, 8), strides=(4, 4), activation='relu', input_shape=self.input_shape))
		model.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
		model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
		model.add(Flatten())
		model.add(Dense(512, activation='relu'))
		model.add(Dense(self.action_size, activation='linear'))
		model.compile(loss=categorical_crossentropy, # TODO: better loss, why categorical?
			optimizer=SGD(lr=self.learning_rate), # TODO: better optimizer, Adam
			metrics=['accuracy'])
		return model

	def process_state(self, state): # TODO: is this needed? can we not input an individual state? if so, can we just input [state]?
		return state.reshape(1, state.shape[0], state.shape[1], state.shape[2])

	def remember(self, state, actions, reward, next_state, done):# remember our information
		self.memory.append((state, actions, reward, next_state, done))# 	Just add it to memory
		
	def act(self, state):
		if self.exploration > np.random.rand():# if we are exploring
			return random.randrange(self.action_size) # random action
		else:
			outcomes = self.model.predict(self.process_state(state)) # choose the action in our current state that returns the highest value
			return np.argmax(outcomes[0])
	
	def train(self, action_num): # train our model
		# training_data = list(self.memory)[len(self.memory)-action_num: len(self.memory)] # take the most recent iterations of the model
		training_data = random.sample(list(self.memory), batch_size)
		terminal_states, terminal_actions, terminal_rewards, nonterminal_states, nonterminal_actions, nonterminal_rewards, nonterminal_next_states = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
		for state, action, reward, next_state, done in training_data:
			if done:
				np.append(terminal_states, state)
				np.append(terminal_actions, action)
				np.append(terminal_rewards, reward)
			else:
				np.append(nonterminal_states, state)
				np.append(nonterminal_actions, action)
				np.append(nonterminal_rewards, reward)
				np.append(nonterminal_next_states, next_state)
		predicted_action_vals = self.model.predict(nonterminal_next_states)
		nonterminal_rewards = nonterminal_rewards + self.discount_rate * np.amax(predicted_action_vals, axis=1) # set the Q function output as: current reward + discount factor * best future reward based on next state
		states = terminal_states + nonterminal_states
		rewards = terminal_rewards + nonterminal_rewards
		actions = terminal_actions + nonterminal_actions
		target_rewards = self.model.predict(states)
		for i in range(batch_size):
			target_rewards[i][actions[i]] = rewards[i][actions[i]]
		# TODO: shuffle
		self.model.fit(states, target_rewards, epochs=1, verbose=0) # fit our model based on our state and target value
		# decay exploration if possible
		if self.exploration > self.exploration_fin:
			self.exploration *= self.exploration_decay


