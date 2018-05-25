from policy_model import PolicyModel
from multi_env import MultiEnv
from a2c_trainer import A2CTrainer, EnvRunner
import env_wrapper
import utils
import tensorflow as tf
import numpy as np
import os
import gym
import time


#--------------------------
# Make an environment for the
# subprocess
#--------------------------
def make_env(rank, env_id="BreakoutNoFrameskip-v4", rand_seed=0):
	def _thunk():
		env = gym.make(env_id)
		env = env_wrapper.NoopResetEnv(env, noop_max=30)
		env = env_wrapper.MaxAndSkipEnv(env, skip=4)
		env.seed(rand_seed + rank)
		env = env_wrapper.EpisodicLifeEnv(env)

		if "FIRE" in env.unwrapped.get_action_meanings():
			env = env_wrapper.FireResetEnv(env)

		env = env_wrapper.WarpFrame(env)
		env = env_wrapper.ClipRewardEnv(env)
		
		return env

	return _thunk


#Parameters---------------------------
n_env = 16
n_step = 5
n_stack = 4
mb_size = n_env * n_step
ent_weight=0.01
value_weight=0.5
max_grad_norm=0.5
lr=7e-4
n_timesteps = 30000000
disp_step = 100
save_step = 500
rand_seed = 0
env_id = "BreakoutNoFrameskip-v4"


#Create multi environment-------------
env = MultiEnv([make_env(i, env_id=env_id) for i in range(n_env)])
save_dir = "./save_" + env_id


#Create a session---------------------
tf.set_random_seed(rand_seed)
np.random.seed(rand_seed)

config = tf.ConfigProto(
	intra_op_parallelism_threads=n_env,
	inter_op_parallelism_threads=n_env
)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


#Create the model---------------------
step_policy = PolicyModel(sess, env.ob_space, env.ac_space, n_env, 1, n_stack)
train_policy = PolicyModel(sess, env.ob_space, env.ac_space, n_env, n_step, n_stack, reuse=True)
trainer = A2CTrainer(
	sess, 
	train_policy, 
	n_env, 
	n_step, 
	ent_weight=ent_weight, 
	value_weight=value_weight, 
	max_grad_norm=max_grad_norm, 
	lr=lr
)
runner = EnvRunner(env, step_policy, n_step, n_stack, gamma=0.99)
utils.show_all_vars()


#Start training----------------------
tf.global_variables_initializer().run(session=sess)

#Load model
saver = tf.train.Saver(var_list=trainer.t_vars, max_to_keep=2)
ckpt = tf.train.get_checkpoint_state(save_dir)
if ckpt:
	print("Loading the model ... ", end="")
	global_step = int(ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1])
	saver.restore(sess, ckpt.model_checkpoint_path)
	print("Done")
else:
	global_step = 0

avg_rewards = []
n_mb = n_timesteps // mb_size
tstart = time.time()

for i in range(global_step, n_mb+global_step+1):
	#Collect data from the environment
	#obs:     (n_env*n_step, nh, nw, nc*n_stack)
	#rewards: (n_env*n_step)
	#actions: (n_env*n_step)
	#values:  (n_env*n_step)
	#masks:   (n_env*(n_step - 1))
	obs, states, rewards, masks, actions, values = runner.run()
	policy_loss, value_loss, policy_entropy = trainer.train(obs, states, rewards, masks, actions, values)
	avg_rewards.append(np.mean(rewards))

	#Compute FPS
	n_sec = time.time() - tstart
	fps = int(((i-global_step)*mb_size) / n_sec)

	#Show the result
	if i % disp_step == 0:
		print("-----------------------------------")
		print("[{:5d} / {:5d}]".format(i, n_mb+global_step))
		print("total_timesteps: {:d}".format(i * mb_size))
		print("fps: {:d}".format(fps))
		print("policy_loss: {:.6f}".format(float(policy_loss)))
		print("policy_entropy: {:.6f}".format(float(policy_entropy)))
		print("value_loss: {:.6f}".format(float(value_loss)))
		print("explained_variance: {:.6f}".format(float(utils.explained_variance(values, rewards))))
		print("avg rewards: {:.6f}".format(sum(avg_rewards)/disp_step))
		print()
		avg_rewards = []

	if i % save_step == 0:
		print("Saving the model ... ", end="")
		saver.save(sess, save_dir+"/model.ckpt", global_step=i)
		print("Done.")
		print()

env.close()
print("Saving the model ... ", end="")
saver.save(sess, save_dir+"/model.ckpt", global_step=i)
print("Done.")
print()