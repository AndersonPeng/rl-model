from multi_env import MultiEnv
from env_runner import MultiEnvRunner
from policy_model import PolicyModel
import env_wrapper
import tensorflow as tf
import numpy as np
import os
import gym
import time
import argparse


#-------------------------
# Make an environment
#-------------------------
def make_env(rank, env_id="CartPole-v0", rand_seed=0):
	def _thunk():
		env = gym.make(env_id)
		env.seed(rand_seed + rank)
		#env = env_wrapper.ClipRewardEnv(env)
		
		return env

	return _thunk


#Parse arguments
#----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--env", default="CartPole-v0")
parser.add_argument("--render", action="store_true")
args = parser.parse_args()


#Parameters
#----------------------------
n_env = 16
n_step = 8
mb_size = n_env*n_step
gamma = 0.99
value_weight = 0.5
ent_weight = 0.05
max_grad_norm=0.5
lr = 7e-4
lr_decay = 0.99
eps = 1e-5
n_iter = 300000
disp_step = 100
save_step = 1000
is_render = args.render
env_id = args.env
save_dir = "./save_" + env_id


#Create multiple environments
#----------------------------
env = MultiEnv([make_env(i, env_id=env_id) for i in range(n_env)])
a_dim = env.ac_space.n
s_dim = env.ob_space.shape[0]
runner = MultiEnvRunner(env, s_dim, n_step, gamma)


#Create the model
#----------------------------
config = tf.ConfigProto(
	intra_op_parallelism_threads=n_env,
	inter_op_parallelism_threads=n_env
)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
policy = PolicyModel(sess, s_dim, a_dim)


#Placeholders
#----------------------------
#action_ph: (mb_size)
#adv_ph:    (mb_size)
#reward_ph: (mb_size)
action_ph = tf.placeholder(tf.int32, [None])
adv_ph = tf.placeholder(tf.float32, [None])
discount_return_ph = tf.placeholder(tf.float32, [None])
lr_ph = tf.placeholder(tf.float32, [])


#Loss
#----------------------------
nll_loss = -policy.cat_dist.log_prob(action_ph)
pg_loss = tf.reduce_mean(adv_ph * nll_loss)
value_loss = tf.reduce_mean(tf.squared_difference(tf.squeeze(policy.value), discount_return_ph) / 2.0)
entropy_bonus = tf.reduce_mean(policy.cat_dist.entropy())
loss = pg_loss + value_weight*value_loss - ent_weight*entropy_bonus


#Optimizer
#----------------------------
t_var = tf.trainable_variables()
grads = tf.gradients(loss, t_var)
grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
grads = list(zip(grads, t_var))
opt = tf.train.RMSPropOptimizer(lr_ph, decay=lr_decay, epsilon=eps).apply_gradients(grads)

tf.contrib.slim.model_analyzer.analyze_vars(t_var, print_info=True)


#Start training
#----------------------------
sess.run(tf.global_variables_initializer())

#Load the model
if not os.path.exists(save_dir):
	os.mkdir(save_dir)

saver = tf.train.Saver(var_list=t_var, max_to_keep=2)
ckpt = tf.train.get_checkpoint_state(save_dir)
if ckpt:
	print("Loading the model ... ", end="")
	global_step = int(ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1])
	saver.restore(sess, ckpt.model_checkpoint_path)
	print("Done.")
else:
	global_step = 0

avg_return = []
return_fp = open(os.path.join(save_dir, "avg_return.txt"), "a+")
t_start = time.time()

for it in range(global_step, n_iter+global_step+1):
	if is_render: env.render()

	#Run the environment
	mb_obs, mb_actions, mb_values, mb_discount_returns = runner.run(policy)
	avg_return.append(np.mean(mb_discount_returns))

	mb_advs = mb_discount_returns - mb_values

	#Train
	cur_pg_loss, cur_value_loss, cur_ent, _ = sess.run([pg_loss, value_loss, entropy_bonus, opt], feed_dict={
		policy.ob_ph: mb_obs,
		action_ph: mb_actions,
		adv_ph: mb_advs,
		discount_return_ph: mb_discount_returns,
		lr_ph: lr
	})

	#Show the result
	if it % disp_step == 0:
		n_sec = time.time() - t_start
		fps = int((it-global_step)*n_env*n_step / n_sec)
		avg_r = sum(avg_return) / disp_step

		print("[{:5d} / {:5d}]".format(it, n_iter))
		print("----------------------------------")
		print("Total timestep = {:d}".format(it * mb_size))
		print("Elapsed time = {:.2f} sec".format(n_sec))
		print("FPS = {:d}".format(fps))
		print("pg_loss = {:.6f}".format(cur_pg_loss))
		print("value_loss = {:.6f}".format(cur_value_loss))
		print("entropy = {:.6f}".format(cur_ent))
		print("Avg return = {:.6f}".format(avg_r))
		print()

		return_fp.write("{:f}\n".format(avg_r))
		return_fp.flush()
		avg_return = []

	#Save
	if it % save_step == 0:
		print("Saving the model ... ", end="")
		saver.save(sess, save_dir+"/model.ckpt", global_step=it)
		print("Done.")
		print()

env.close()
return_fp.close()
print("Saving the model ... ", end="")
saver.save(sess, save_dir+"/model.ckpt", global_step=n_iter+global_step)
print("Done.")
print()