from multi_env import MultiEnv
from env_runner import MultiEnvRunner
from policy_model import PolicyModel
from discriminator_model import DiscriminatorModel
import env_wrapper
import tensorflow as tf
import numpy as np
import os
import sys
import gym
import time
import argparse
import pickle


#-------------------------
# Make an environment
#-------------------------
def make_env(rank, env_id="CartPole-v0", rand_seed=0, unwrap=False):
	def _thunk():
		env = gym.make(env_id)
		if unwrap: env = env.unwrapped
		env.seed(rand_seed + rank)
		
		return env

	return _thunk


#Parse arguments
#----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--env", default="CartPole-v0")
parser.add_argument("--render", action="store_true")
parser.add_argument("--unwrap", action="store_true")
args = parser.parse_args()


#Parameters
#----------------------------
n_env = 8
n_step = 128
mb_size = n_env*n_step
sample_mb_size = 64
sample_n_mb = mb_size // sample_mb_size
sample_n_epoch = 4
gamma = 0.99
lamb = 0.95
clip_val = 0.2
ent_weight = 0.01
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
expert_traj_dir = "../../ppo/discrete/save_" + env_id


#Load the expert trajectories
#----------------------------
expert_traj_filename = os.path.join(expert_traj_dir, "traj.pkl")

if os.path.exists(expert_traj_filename):
	expert_traj = pickle.load(open(expert_traj_filename, "rb"))
else:
	print("ERROR: No expert trajectory file found")
	sys.exit(1)


#Create multiple environments
#----------------------------
env = MultiEnv([make_env(i, env_id=env_id, unwrap=args.unwrap) for i in range(n_env)])
s_dim = env.ob_space.shape[0]
a_dim = env.ac_space.n
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
policy = PolicyModel(sess, s_dim, a_dim, "policy")
dis = DiscriminatorModel(sess, s_dim, a_dim, name="discriminator")


#Placeholders
#----------------------------
#action_ph:          (mb_size)
#old_neg_logprob_ph: (mb_size)
#old_v_pred_ph:      (mb_size)
#adv_ph:             (mb_size)
#return_ph:          (mb_size)
action_ph = tf.placeholder(tf.int32, [None], name="action")
old_neg_logprob_ph = tf.placeholder(tf.float32, [None], name="old_negtive_log_prob")
old_v_pred_ph = tf.placeholder(tf.float32, [None], name="old_value_pred")
adv_ph = tf.placeholder(tf.float32, [None], name="advantage")
return_ph = tf.placeholder(tf.float32, [None], name="return")
lr_ph = tf.placeholder(tf.float32, [])
clip_ph = tf.placeholder(tf.float32, [])


#PPO Loss
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


#GAIL Loss
#----------------------------
dis_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
	logits=dis.logit_real,
	labels=tf.ones_like(dis.prob_real)
))
dis_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
	logits=dis.logit_fake,
	labels=tf.zeros_like(dis.prob_fake)
))
dis_loss = dis_loss_fake + dis_loss_real


#Optimizer
#----------------------------
t_var = tf.trainable_variables()
ppo_var = [var for var in t_var if "policy" in var.name]
dis_var = [var for var in t_var if "discriminator" in var.name]

grads = tf.gradients(loss, ppo_var)
grads, actor_grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
grads = list(zip(grads, ppo_var))
ppo_opt = tf.train.AdamOptimizer(lr_ph, epsilon=eps).apply_gradients(grads)
dis_opt = tf.train.AdamOptimizer(lr_ph, epsilon=eps).minimize(dis_loss, var_list=dis_var)

tf.contrib.slim.model_analyzer.analyze_vars(ppo_var, print_info=True)
tf.contrib.slim.model_analyzer.analyze_vars(dis_var, print_info=True)


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

expert_traj = np.concatenate([np.array(t, dtype=np.float32) for t in expert_traj], 0)
np.random.shuffle(expert_traj)
mean_returns = []
std_returns  = []
rand_idx = np.arange(mb_size)
return_fp = open(os.path.join(save_dir, "avg_return.txt"), "a+")
t_start = time.time()

for it in range(global_step, n_iter+global_step+1):
	if is_render: env.render()

	#Run the environment
	mb_obs, mb_actions, mb_neg_logprobs, mb_values, mb_returns = runner.run(policy, dis)

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

			cur_pg_loss, cur_v_loss, cur_ent, _ = sess.run([pg_loss, v_loss, ent, ppo_opt], feed_dict={
				policy.ob_ph: sample_obs,
				action_ph: sample_actions,
				old_neg_logprob_ph: sample_neg_logprobs,
				old_v_pred_ph: sample_values,
				adv_ph: sample_advs,
				return_ph: sample_returns,
				lr_ph: lr,
				clip_ph: clip_val
			})

	#Train Discriminator
	np.random.shuffle(rand_idx)

	for j in range(sample_n_mb):
		sample_idx = rand_idx[j*sample_mb_size : (j+1)*sample_mb_size]
		sample_obs = mb_obs[sample_idx]
		sample_actions = mb_actions[sample_idx]

		sample_actions_onehot = np.zeros([sample_mb_size, a_dim])
		for k in range(sample_mb_size):
			sample_actions_onehot[k, sample_actions[k]] = 1

		traj_real = expert_traj[np.random.randint(0, expert_traj.shape[0], sample_mb_size), :]
		traj_fake = np.concatenate([sample_obs, sample_actions_onehot], 1)

		cur_dis_loss, _ = sess.run([dis_loss, dis_opt], feed_dict={
			dis.traj_fake_ph: traj_fake,
			dis.traj_real_ph: traj_real,
			lr_ph: lr
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
		print("dis_loss = {:.6f}".format(cur_dis_loss))
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