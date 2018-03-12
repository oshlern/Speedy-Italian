import keras

class QNet:
	# Define Q neural network
	def __init__(self, state_shape, action_size):
		self.state_size = reduce(mul, state_shape, 1)
		self.input_shape = state_shape
		self.action_size = action_size
		self.memory = deque(maxlen=2000)
		self.discount_rate = 0.99
		self.exploration = 1.0
		# self.exploration_decay = 0.99954
		self.exploration_decay = 0.99
		self.exploration_min = 0.05
		self.learning_rate = 0.9
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
		model.compile(loss=categorical_crossentropy,
			optimizer=SGD(lr=0.08),
			metrics=['accuracy'])
		return model

	def process_state(self, state):
		return state.reshape(1, state.shape[0], state.shape[1], state.shape[2])

	def remember(self, state, action, reward, next_state, done):# remember our information
		self.memory.append((state, action, reward, next_state, done))# 	Just add it to memory
		
	def act(self, state):
		if self.exploration > np.random.rand():# if we are exploring
			return random.randrange(self.action_size) # random action
		else:
			outcomes = self.model.predict(self.process_state(state)) # choose the action in our current state that returns the highest value
			return np.argmax(outcomes[0])
	
	def train(self, action_num): # train our model
		# training_data = list(self.memory)[len(self.memory)-action_num: len(self.memory)] # take the most recent iterations of the model
		training_data = random.sample(list(self.memory), batch_size)
		for state, action, reward, next_state, done in training_data: # cycling through every moment that we are referencing
			true_reward = reward # define the model expected output as the current reward,
			if not done:
				predicted_now = self.model.predict(self.process_state(state))[0][action]
				predicted_next = self.model.predict(self.process_state(next_state))[0][action]
				true_reward = predicted_now + self.learning_rate * (reward + self.discount_rate * predicted_next - predicted_now) # set the Q function output as: current reward + timerate * best action based on next state
			predicted_rewards = self.model.predict(self.process_state(state)) # set the reward for our action in this state to the reward we just got
			predicted_rewards[0][action] = true_reward
			self.model.fit(self.process_state(state), predicted_rewards, epochs=1, verbose=0) # fit our model based on our state and target value
		# decay exploration if possible
		if self.exploration > self.exploration_min:
            self.exploration *= self.exploration_decay