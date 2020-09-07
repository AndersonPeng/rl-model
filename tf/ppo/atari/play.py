from policy_model import PolicyModel
import env_wrapper
import tensorflow as tf
import numpy as np
import os
import gym
import time
import argparse


#Parse arguments
#----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--env", default="BreakoutNoFrameskip-v4")
args = parser.parse_args()


#Parameters
#----------------------------
n_stack = 4
env_id = args.env
save_dir = "./save_" + env_id


#Create multiple environments
#----------------------------
env = gym.make(env_id)
env = env_wrapper.NoopResetEnv(env, noop_max=30)
env = env_wrapper.MaxAndSkipEnv(env, skip=4)
env = env_wrapper.EpisodicLifeEnv(env)

if "FIRE" in env.unwrapped.get_action_meanings():
	env = env_wrapper.FireResetEnv(env)

env = env_wrapper.WarpFrame(env)
env = env_wrapper.FrameStack(env, n_stack)

a_dim = env.action_space.n
img_height, img_width, c_dim = env.observation_space.shape


#Create the model
#----------------------------
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
policy = PolicyModel(sess, img_height, img_width, c_dim, a_dim)


#Start training
#----------------------------
sess.run(tf.global_variables_initializer())

#Load the model
if not os.path.exists(save_dir):
	os.mkdir(save_dir)

saver = tf.train.Saver(max_to_keep=2)
ckpt = tf.train.get_checkpoint_state(save_dir)
if ckpt:
	print("Loading the model ... ", end="")
	saver.restore(sess, ckpt.model_checkpoint_path)
	print("Done.")

for it in range(100):
	ob = env.reset()
	total_reward = 0

	while True:
		env.render()
		action = policy.action_step(np.expand_dims(ob.__array__(), axis=0))
		ob, reward, done, info = env.step(action[0])
		total_reward += reward

		if done:
			print("total_reward = {}".format(total_reward))
			break