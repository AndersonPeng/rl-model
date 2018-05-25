from policy_model import PolicyModel
import tensorflow as tf
import numpy as np
import utils
import env_wrapper
import os
import gym
import time


#Parameters-----------------------------
n_stack = 4
env_id = "BreakoutNoFrameskip-v4"


#Create an environment------------------
env = gym.make(env_id)
env = env_wrapper.NoopResetEnv(env, noop_max=30)
env = env_wrapper.MaxAndSkipEnv(env, skip=4)
env = env_wrapper.EpisodicLifeEnv(env)

if "FIRE" in env.unwrapped.get_action_meanings():
	env = env_wrapper.FireResetEnv(env)

env = env_wrapper.WarpFrame(env)
env = env_wrapper.FrameStack(env, 4)
save_dir = "./save_" + env_id


#Create a session---------------------
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


#Create the model---------------------
step_policy = PolicyModel(sess, env.observation_space, env.action_space, n_env=1, n_step=1, n_stack=1)

#Load model
saver = tf.train.Saver(var_list=utils.find_trainable_vars("policy_model"), max_to_keep=2)
ckpt = tf.train.get_checkpoint_state(save_dir)
if ckpt:
	print("Loading the model ... ", end="")
	global_step = int(ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1])
	saver.restore(sess, ckpt.model_checkpoint_path)
	print("Done")
else:
	raise NotImplementedError


#Start playing-----------------------
obs = env.reset()
score = 0

while True:
	#obs: (1, img_height, img_width, channel_dim*n_stack)
	obs = np.expand_dims(obs.__array__(), axis=0)
	a, v, _ = step_policy.step(obs)
	obs, reward, done, info = env.step(a)
	env.render()
	score += reward

	if done:
		env.reset()
		print("score: {}".format(score))
		score = 0

	#time.sleep(0.01)