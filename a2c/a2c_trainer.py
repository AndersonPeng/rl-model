import numpy as np
import tensorflow as tf

import utils


#A2C Trainer
class A2CTrainer():
	#--------------------------
	# Constructor
	#--------------------------
	def __init__(
		self,
		sess,
		train_policy,
		n_env,
		n_step,
		ent_weight=0.01,
		value_weight=0.5,
		max_grad_norm=0.5,
		lr=7e-4,
		lr_decay=0.99,
		eps=1e-5
	):
		#Config--------------------------------
		self.sess = sess
		self.train_policy = train_policy
		self.n_env = n_env
		self.n_step = n_step
		self.lr = lr
		mb_size = n_env * n_step
		

		#Placeholders--------------------------
		#action_ph: (mb_size)
		#adv_ph:    (mb_size)
		#reward_ph: (mb_size)
		self.action_ph = tf.placeholder(tf.int32, [mb_size])
		self.adv_ph = tf.placeholder(tf.float32, [mb_size])
		self.reward_ph = tf.placeholder(tf.float32, [mb_size])
		self.lr_ph = tf.placeholder(tf.float32, [])


		#Loss----------------------------------
		nll_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.train_policy.pi, labels=self.action_ph)
		self.pg_loss = tf.reduce_mean(self.adv_ph * nll_loss)
		self.value_loss = tf.reduce_mean(tf.squared_difference(tf.squeeze(self.train_policy.value), self.reward_ph) / 2.0)
		self.entropy_bonus = tf.reduce_mean(utils.cat_entropy(self.train_policy.pi))
		self.loss = self.pg_loss + value_weight*self.value_loss - ent_weight*self.entropy_bonus


		#Optimizer-----------------------------
		self.t_vars = utils.find_trainable_vars("policy_model")
		grads = tf.gradients(self.loss, self.t_vars)

		#Clip gradient norm
		if max_grad_norm is not None:
			grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

		grads = list(zip(grads, self.t_vars))
		self.opt = tf.train.RMSPropOptimizer(self.lr_ph, decay=lr_decay, epsilon=eps).apply_gradients(grads)


	#--------------------------
	# Train (for a batch)
	#--------------------------
	def train(self, mb_ob, mb_state, mb_reward, mb_mask, mb_action, mb_value):
		#mb_ob:     (n_env*n_step, nh, nw, nc*n_stack)
		#mb_reward: (n_env*n_step)
		#mb_action: (n_env*n_step)
		#mb_value:  (n_env*n_step)
		#mb_mask:   (n_env*(n_step - 1))
		#mb_adv:    (n_env*n_step)
		mb_adv = mb_reward - mb_value

		feed_dict = {
			self.train_policy.ob_ph: mb_ob,
			self.action_ph: mb_action,
			self.adv_ph: mb_adv,
			self.reward_ph: mb_reward,
			self.lr_ph: self.lr
		}
		if mb_state != []:
			feed_dict[self.train_policy.state_ph] = mb_state
			feed_dict[self.train_policy.mask_ph] = mb_mask

		cur_pg_loss, cur_value_loss, cur_entropy_bonus, _ = self.sess.run(
			[self.pg_loss, self.value_loss, self.entropy_bonus, self.opt],
			feed_dict
		)
		return cur_pg_loss, cur_value_loss, cur_entropy_bonus


#Multi-Environment Runner
class EnvRunner():
	#--------------------------
	# Constructor
	#--------------------------
	def __init__(self, env, policy, n_step=5, n_stack=4, gamma=0.99):
		img_height, img_width, channel_dim = env.ob_space.shape

		self.env = env
		self.policy = policy
		self.n_env = env.n_env
		self.n_step = n_step
		self.channel_dim = channel_dim
		self.mb_ob_shape = (self.n_env*self.n_step, img_height, img_width, channel_dim*n_stack)
		self.gamma = gamma

		#obs: (n_env, img_height, img_width, channel_dim*n_stack)
		#dones: (n_env)
		#states: []
		self.obs =  np.zeros((self.n_env, img_height, img_width, channel_dim*n_stack), dtype=np.uint8)
		self.dones = [False for _ in range(self.n_env)]
		self.states = policy.init_state
		
		obs = env.reset()
		self.update_obs(obs)


	#--------------------------
	# Update observations
	# Do frame-stacking
	#--------------------------
	def update_obs(self, obs):
		#Shift 1 frame in the stack
		#Then put 1 new frame into the stack
		self.obs = np.roll(self.obs, shift=-self.channel_dim, axis=3)
		self.obs[:, :, :, -self.channel_dim:] = obs


	#--------------------------
	# Run the environment n steps
	# (for a batch)
	#--------------------------
	def run(self):
		mb_ob = []
		mb_reward = []
		mb_action = []
		mb_value = []
		mb_done = []
		mb_state = self.states


		#1. Run n steps---------------------------
		for n in range(self.n_step):
			#actions: (n_env)
			#values:  (n_env)
			actions, values, states = self.policy.step(self.obs, self.states, self.dones)
			mb_ob.append(np.copy(self.obs))
			mb_action.append(actions)
			mb_value.append(values)
			mb_done.append(self.dones)

			#obs:     (n_env, img_height, img_width, channel_dim)
			#rewards: (n_env)
			#dones:   (n_env)
			obs, rewards, dones, _ = self.env.step(actions)
			self.states = states
			self.dones = dones

			for i, done in enumerate(dones):
				if done:
					self.obs[i] = self.obs[i] * 0

			self.update_obs(obs)
			mb_reward.append(rewards)

		#Put the prev dones
		mb_done.append(self.dones)


		#2. Batch of steps to batch of rollouts---
		#mb_ob:     (n_env*n_step, img_height, img_width, channel_dim*n_stack)
		#mb_reward: (n_env, n_step)
		#mb_action: (n_env, n_step)
		#mb_value:  (n_env, n_step)
		#mb_done:   (n_env, n_step - 1)
		#mb_mask:   (n_env, n_step - 1)
		mb_ob = np.asarray(mb_ob, dtype=np.uint8).swapaxes(1, 0).reshape(self.mb_ob_shape)
		mb_reward = np.asarray(mb_reward, dtype=np.float32).swapaxes(1, 0)
		mb_action = np.asarray(mb_action, dtype=np.int32).swapaxes(1, 0)
		mb_value = np.asarray(mb_value, dtype=np.float32).swapaxes(1, 0)
		mb_done = np.asarray(mb_done, dtype=np.bool).swapaxes(1, 0)
		mb_mask = mb_done[:, :-1]
		mb_done = mb_done[:, 1:]

		#last_values: (n_env)
		last_values = self.policy.value_step(self.obs, self.states, self.dones).tolist()


		#3. Compute discounted value---------------
		#For each env
		for n, (rewards, dones, value) in enumerate(zip(mb_reward, mb_done, last_values)):
			#rewards: (n_step)
			#dones:   (n_step - 1)
			rewards = rewards.tolist()
			dones = dones.tolist()

			#The last step is not done, add the last value
			if dones[-1] == 0:
				rewards = utils.discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
			
			#The last step is done, don't add the last value
			else:
				rewards = utils.discount_with_dones(rewards, dones, self.gamma)

			mb_reward[n] = rewards

		#mb_obs:     (n_env*n_step, img_height, img_width, channel_dim*n_stack)
		#mb_rewards: (n_env*n_step)
		#mb_actions: (n_env*n_step)
		#mb_values:  (n_env*n_step)
		#mb_masks:   (n_env*(n_step - 1))
		return mb_ob, mb_state, mb_reward.flatten(), mb_mask.flatten(), mb_action.flatten(), mb_value.flatten()