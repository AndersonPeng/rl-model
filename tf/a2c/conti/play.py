from policy_model import PolicyModel
import tensorflow as tf
import numpy as np
import os
import gym
import argparse


#Parse arguments
#----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--env", default="BipedalWalker-v2")
parser.add_argument("--unwrap", action="store_true")
args = parser.parse_args()


#Parameters
#----------------------------
env_id = args.env
save_dir = "./save_" + env_id


#Create the environment
#----------------------------
env = gym.make(env_id)
if args.unwrap: env = env.unwrapped
a_dim = env.action_space.shape[0]
a_low = env.action_space.low[0]
a_hight = env.action_space.high[0]
s_dim = env.observation_space.shape[0]


#Create the model
#----------------------------
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
policy = PolicyModel(sess, s_dim, a_dim, a_low, a_hight)


#Start playing
#----------------------------
sess.run(tf.global_variables_initializer())

#Load the model
saver = tf.train.Saver(max_to_keep=2)
ckpt = tf.train.get_checkpoint_state(save_dir)
if ckpt:
	print("Loading the model ... ", end="")
	saver.restore(sess, ckpt.model_checkpoint_path)
	print("Done.")

logstd = np.zeros((1, a_dim), dtype=np.float32)
logstd.fill(-6.0)

for it in range(100):
	ob = env.reset()
	total_reward = 0

	while True:
		env.render()
		action = policy.action_step(np.expand_dims(ob.__array__(), axis=0), logstd)
		ob, reward, done, info = env.step(action[0])
		total_reward += reward

		if done:
			print("total_reward = {}".format(total_reward))
			break