# PYTHON 3 (sadly)
import numpy
import gym
import time
# import gym_pull
# gym_pull.pull('github.com/ppaquette/gym-super-mario')        # Only required once, envs will be loaded with import gym_pull afterwards
# env = gym.make('ppaquette/SuperMarioBros-1-1-v0')
from model import *

def addToFile(file, what): # from https://stackoverflow.com/questions/13203868/how-to-write-to-csv-and-not-overwrite-past-text
	f = csv.writer(open(file, 'a')).writerow(what) # appends to csv file

if __name__ == "__main__": # Main part of game:
	env = gym.make('CartPole-v0')
	state_shape = env.observation_space.shape
	action_size = env.action_space.n
	done = False
	batch_size = 32
	EPISODES = 100000
	render_rate = 500
	print_rate = 20
	hyperparams={'max_len': 2000000, 'discount_rate': 0.99, 'exploration_init': 1.0, 'exploration_fin': 0.005, 'exploration_decay': 0.99993, 'learning_rate': 0.02, 'batch_size': 32, 'update_target_freq': 160}
	agent = QNet(state_shape, action_size, layer_sizes=[], hyperparams=hyperparams)

	for e in range(EPISODES):
		render = e%render_rate == 0
		state = env.reset()
		score = 0
		while True:
			if render:
				time.sleep(1/30)
				env.render()
				action = agent.test_act(state)
			else:
				action = agent.act(state)
			next_state, reward, done, _ = env.step(action)
			reward = reward if not done else -10
			if not done:
				score += reward # Add your reward to the score
			agent.remember(state, action, reward, next_state, done)
			state = next_state
			if done:
				if e%print_rate == 0:
					print("episode: {}/{}, score: {}, e: {:.2}"
						.format(e, EPISODES, score, agent.exploration))
				addToFile("test.csv",([e, score])) # add data to file for later analyzation
				break
		if len(agent.memory) > batch_size:
			agent.train(batch_size)
	model_json = agent.model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)

