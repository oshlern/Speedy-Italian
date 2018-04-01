# PYTHON 3 (sadly)
import numpy
import gym
import time
import math
from model import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('fivethirtyeight')
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def addToFile(file, what): # from https://stackoverflow.com/questions/13203868/how-to-write-to-csv-and-not-overwrite-past-text
	f = csv.writer(open(file, 'a')).writerow(what) # appends to csv file

def animate(i, argu):
    graph_data = open(argu,'r').read()
    lines = graph_data.split('\n')
    xs = []
    ys = []
    for line in lines:
        if len(line) > 1:
            x, y = line.split(',')
            xs.append(x)
            ys.append(y)
    ax1.clear()
    ax1.plot(xs, ys)

if __name__ == "__main__": # Main part of game:
	filename = "runs/" + str(math.ceil(time.time())) + ".csv"
	addToFile(filename, "")
	env = gym.make('CartPole-v0')
	state_shape = env.observation_space.shape
	action_size = env.action_space.n
	done = False
	batch_size = 32
	EPISODES = 100000
	render_rate = 500
	print_rate = 50
	visualization_state = False
	hyperparams={'max_len': 2000000, 'discount_rate': 0.99, 'exploration_init': 1.0, 'exploration_fin': 0.009, 'exploration_decay': 0.9997, 'learning_rate': 0.015, 'batch_size': 32, 'update_target_freq': 1000}
	agent = QNet(state_shape, action_size, layer_sizes=[4], hyperparams=hyperparams)
	sample_weights = [np.array([[-0.01, 0.01], [0.0,0.0], [-3,3], [-0.5,0.5]]), np.array([0.0,0.0])]
	# agent.model.set_weights(sample_weights)
	# for e in range(10):
	# 	state = env.reset()
	# 	while True:
	# 		time.sleep(1/30)
	# 		env.render()
	# 		action = agent.test_act(state)
	# 		next_state, reward, done, _ = env.step(action)
	# 		state = next_state
	# 		if done:
	# 			break
	for e in range(EPISODES):
		if visualization_state:
			ani = animation.FuncAnimation(fig, animate, fargs=[filename], interval=1000)
			plt.show(block=False)
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
					addToFile(filename,([e, score])) # add data to file for later analyzation
				break
		if len(agent.memory) > batch_size:
			agent.train(batch_size)
	model_json = agent.model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)

