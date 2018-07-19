from policy_model import PolicyModel
from discriminator_model import DiscriminatorModel
import tensorflow as tf
import numpy as np
import gym
import os
import sys
import time
import argparse
import pickle


#Parse arguments
#----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--env", default="BipedalWalker-v2")
args = parser.parse_args()


#Parameters
#----------------------------
n_epoch = 32
mb_size = 128
lr = 1e-4
lr_decay = 0.99
eps = 1e-5
disp_step = 1000
save_step = 10000
env_id = args.env
save_dir = "./save_" + env_id
expert_traj_dir = "../expert_ppo/conti/save_" + env_id


#Create the environment
#----------------------------
env = gym.make(env_id)
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_low = env.action_space.low[0]
a_high = env.action_space.high[0]


#Create the model
#----------------------------
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
policy = PolicyModel(sess, s_dim, a_dim, a_low, a_high, name="policy")
dis = DiscriminatorModel(sess, s_dim, a_dim, name="discriminator")


#Placeholders
#----------------------------
#action_ph: (mb_size, a_dim)
action_ph = tf.placeholder(tf.float32, [None, a_dim], name="action")
lr_ph = tf.placeholder(tf.float32, [])


#Loss
#----------------------------
loss = tf.reduce_mean(policy.distrib.neg_logp(action_ph))


#Optimizer
#----------------------------
t_var = tf.trainable_variables()
policy_var = [var for var in t_var if "policy/actor" in var.name]
opt = tf.train.AdamOptimizer(lr_ph, epsilon=eps).minimize(loss, var_list=policy_var)

tf.contrib.slim.model_analyzer.analyze_vars(policy_var, print_info=True)


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

#Load the expert trajectories
expert_traj_filename = os.path.join(expert_traj_dir, "traj.pkl")

if os.path.exists(expert_traj_filename):
	expert_traj = pickle.load(open(expert_traj_filename, "rb"))
else:
	print("ERROR: No expert trajectory file found")
	sys.exit(1)

expert_traj = np.concatenate([np.array(t, dtype=np.float32) for t in expert_traj], 0)
np.random.shuffle(expert_traj)

total_rewards = np.zeros(8)
total_lens = np.zeros(8)
it = 0
n_mb = expert_traj.shape[0]//mb_size

for i_epoch in range(n_epoch):
	#Train
	for i in range(n_mb):
		mb_traj = expert_traj[np.random.randint(0, expert_traj.shape[0], mb_size), :]
		mb_obs = mb_traj[:, :s_dim]
		mb_actions = mb_traj[:, s_dim:]

		cur_loss, _ = sess.run([loss, opt], feed_dict={
			policy.ob_ph: mb_obs,
			action_ph: mb_actions,
			lr_ph: lr
		})

		#Show the result
		if it % disp_step == 0:
			for j in range(8):
				ob = env.reset()
				total_reward = 0
				total_len = 0

				while True:
					action = policy.action_step(np.expand_dims(ob.__array__(), axis=0))
					ob, reward, done, info = env.step(action[0])
					total_reward += reward
					total_len += 1

					if done:
						total_rewards[j] = total_reward
						total_lens[j] = total_len
						break

			print("[{:5d} / {:5d}] [{:5d} / {:5d}]".format(i_epoch, n_epoch, i, n_mb))
			print("----------------------------------")
			print("loss = {:.6f}".format(cur_loss))
			print("mean_total_reward = {:.6f}".format(total_rewards.mean()))
			print("mean_len = {:.2f}".format(total_lens.mean()))
			print()

		#Save
		if it % save_step == 0:
			print("Saving the model ... ", end="")
			saver.save(sess, save_dir+"/model.ckpt", global_step=0)
			print("Done.")
			print()

		it += 1

env.close()
print("Saving the model ... ", end="")
saver.save(sess, save_dir+"/model.ckpt", global_step=0)
print("Done.")
print()