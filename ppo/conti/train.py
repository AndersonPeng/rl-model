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
def make_env(rank, env_id="BipedalWalker-v2", rand_seed=0, unwrap=False):
	def _thunk():
		env = gym.make(env_id)
		if unwrap: env = env.unwrapped
		env.seed(rand_seed + rank)
		
		return env

	return _thunk


#Parse arguments
#----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--env", default="BipedalWalker-v2")
parser.add_argument("--render", action="store_true")
parser.add_argument("--unwrap", action="store_true")
args = parser.parse_args()


#Parameters
#----------------------------
n_env = 1
n_step = 2048
mb_size = n_env*n_step
sample_mb_size = 64
sample_n_mb = mb_size // sample_mb_size
sample_n_epoch = 10
gamma = 0.99
lamb = 0.95
clip_val = 0.2
ent_weight = 0.0
v_weight = 0.5
max_grad_norm = 0.5
lr = 3e-4
lr_decay = 0.99
eps = 1e-5
n_iter = 30000
disp_step = 10
save_step = 100
is_render = args.render
env_id = args.env
save_dir = "./save_" + env_id


#Create multiple environments
#----------------------------
env = MultiEnv([make_env(i, env_id=env_id) for i in range(n_env)])
a_dim = env.ac_space.shape[0]
s_dim = env.ob_space.shape[0]
a_low = env.ac_space.low[0]
a_high = env.ac_space.high[0]
runner = MultiEnvRunner(env, s_dim, a_dim, n_step, gamma, lamb)


#Create the model
#----------------------------
config = tf.ConfigProto(
	allow_soft_placement=True,
	intra_op_parallelism_threads=n_env,
	inter_op_parallelism_threads=n_env
)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
policy = PolicyModel(sess, s_dim, a_dim, a_low, a_high, "policy")


#Placeholders
#----------------------------
#action_ph:          (mb_size, a_dim)
#old_neg_logprob_ph: (mb_size)
#old_v_pred_ph:      (mb_size)
#adv_ph:             (mb_size)
#return_ph:          (mb_size)
action_ph = tf.placeholder(tf.float32, [None, a_dim], name="action")
old_neg_logprob_ph = tf.placeholder(tf.float32, [None], name="old_negtive_log_prob")
old_v_pred_ph = tf.placeholder(tf.float32, [None], name="old_value_pred")
adv_ph = tf.placeholder(tf.float32, [None], name="advantage")
return_ph = tf.placeholder(tf.float32, [None], name="return")
lr_ph = tf.placeholder(tf.float32, [])
clip_ph = tf.placeholder(tf.float32, [])


#Loss
#----------------------------
neg_logprob = policy.distrib.neg_logp(action_ph)
ent = tf.reduce_mean(policy.distrib.entropy())

v_pred = policy.value
v_pred_clip = old_v_pred_ph + tf.clip_by_value(v_pred - old_v_pred_ph, -clip_ph, clip_ph)
v_loss1 = tf.square(v_pred - return_ph)
v_loss2 = tf.square(v_pred_clip - return_ph)
v_loss = 0.5 * tf.reduce_mean(tf.maximum(v_loss1, v_loss2))

ratio = tf.exp(old_neg_logprob_ph - neg_logprob)
pg_loss1 = -adv_ph * ratio
pg_loss2 = -adv_ph * tf.clip_by_value(ratio, 1.0 - clip_ph, 1.0 + clip_ph)
pg_loss = tf.reduce_mean(tf.maximum(pg_loss1, pg_loss2))

loss = pg_loss - ent_weight*ent + v_weight*v_loss


#Optimizer
#----------------------------
t_var = tf.trainable_variables()

grads = tf.gradients(loss, t_var)
grads, actor_grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
grads = list(zip(grads, t_var))
opt = tf.train.AdamOptimizer(lr_ph, epsilon=eps).apply_gradients(grads)

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

mean_returns = []
std_returns  = []
rand_idx = np.arange(mb_size)
return_fp = open(os.path.join(save_dir, "avg_return.txt"), "a+")
t_start = time.time()

for it in range(global_step, n_iter+global_step+1):
	if is_render: env.render()

	#Run the environment
	mb_obs, mb_actions, mb_neg_logprobs, mb_values, mb_returns = runner.run(policy)

	#Train
	for i in range(sample_n_epoch):
		np.random.shuffle(rand_idx)

		for j in range(sample_n_mb):
			sample_idx = rand_idx[j*sample_mb_size : (j+1)*sample_mb_size]
			sample_obs = mb_obs[sample_idx]
			sample_actions = mb_actions[sample_idx]
			sample_values = mb_values[sample_idx]
			sample_neg_logprobs = mb_neg_logprobs[sample_idx]
			sample_returns = mb_returns[sample_idx]
			sample_advs = sample_returns - sample_values
			sample_advs = (sample_advs - sample_advs.mean()) / (sample_advs.std() + 1e-8)

			cur_pg_loss, cur_v_loss, cur_ent, _ = sess.run([pg_loss, v_loss, ent, opt], feed_dict={
				policy.ob_ph: sample_obs,
				action_ph: sample_actions,
				old_neg_logprob_ph: sample_neg_logprobs,
				old_v_pred_ph: sample_values,
				adv_ph: sample_advs,
				return_ph: sample_returns,
				lr_ph: lr,
				clip_ph: clip_val
			})

	#Show the result
	if it % disp_step == 0 and it > global_step:
		n_sec = time.time() - t_start
		fps = int((it-global_step)*n_env*n_step / n_sec)
		mean_return, std_return, mean_len = runner.get_performance()
		mean_returns.append(mean_return)
		std_returns.append(std_return)

		print("[{:5d} / {:5d}]".format(it, n_iter+global_step))
		print("----------------------------------")
		print("Total timestep = {:d}".format(it * mb_size))
		print("Elapsed time = {:.2f} sec".format(n_sec))
		print("FPS = {:d}".format(fps))
		print("pg_loss = {:.6f}".format(cur_pg_loss))
		print("v_loss = {:.6f}".format(cur_v_loss))
		print("entropy = {:.6f}".format(cur_ent))
		print("mean_total_reward = {:.6f}".format(mean_return))
		print("mean_len = {:.2f}".format(mean_len))
		print()

	#Save
	if it % save_step == 0 and it > global_step:
		print("Saving the model ... ", end="")
		saver.save(sess, save_dir+"/model.ckpt", global_step=it)

		for mean, std in zip(mean_returns, std_returns):
			return_fp.write("{:f},{:f}\n".format(mean, std))
		
		return_fp.flush()
		mean_returns.clear()
		std_returns.clear()
		print("Done.")
		print()

env.close()
return_fp.close()
print("Saving the model ... ", end="")
saver.save(sess, save_dir+"/model.ckpt", global_step=n_iter+global_step)
print("Done.")
print()