import tensorflow as tf
import numpy as np
import ops


class PolicyModel(object):
	#---------------------------
	# Constructor
	#---------------------------
	def __init__(self, sess, s_dim, a_dim, a_low, a_high, reuse=False):
		self.sess = sess

		with tf.variable_scope("actor", reuse=reuse):
			#ob_ph: (mb_size, s_dim)
			self.ob_ph = tf.placeholder(tf.float32, (None, s_dim))

			#fc1: (mb_size, 256)
			h = ops.fc(self.ob_ph, 256, name="acotr_fc1")
			h = tf.nn.relu(h)
			
			#fc2: (mb_size, 128)
			h = ops.fc(h, 128, name="actor_fc2")
			h = tf.nn.relu(h)

			#pi_mu: (mb_size, a_dim)
			mu = ops.fc(h, a_dim, name="actor_fc_mu")
			mu = tf.nn.tanh(mu)

			#pi_sigma: (mb_size, a_dim)
			sigma = ops.fc(h, a_dim, name="actor_fc_sigma")
			sigma = tf.nn.softplus(sigma)

		with tf.variable_scope("critic", reuse=reuse):
			#fc1: (mb_size, 256)
			h = ops.fc(self.ob_ph, 256, name="critic_fc1")
			h = tf.nn.relu(h)
			
			#fc2: (mb_size, 128)
			h = ops.fc(h, 128, name="critic_fc2")
			h = tf.nn.relu(h)

			#value: (mb_size, 1)
			value = ops.fc(h, 1, name="critic_fc_value")
		
		with tf.name_scope("wrap_a_out"):
			mu = mu * a_high
			sigma = sigma + 1e-5

		#value:  (mb_size)
		#action: (mb_size)
		self.value = value[:, 0]
		self.normal_dist = tf.distributions.Normal(mu, sigma)
		self.action = tf.clip_by_value(tf.squeeze(self.normal_dist.sample(1), axis=0), a_low, a_high)


	#---------------------------
	# Forward step
	#---------------------------
	def step(self, mb_obs):
		a, v = self.sess.run([self.action, self.value], {self.ob_ph: mb_obs})
		return a, v


	#---------------------------
	# Forward step for 
	# value function
	#---------------------------
	def value_step(self, mb_obs):
		return self.sess.run(self.value, {self.ob_ph: mb_obs})