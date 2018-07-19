import tensorflow as tf
import numpy as np
import ops
import distribs


class PolicyModel(object):
	#---------------------------
	# Constructor
	#---------------------------
	def __init__(self, sess, s_dim, a_dim, name="policy", reuse=False):
		self.sess = sess

		with tf.variable_scope(name, reuse=reuse):
			#ob_ph: (mb_size, s_dim)
			self.ob_ph = tf.placeholder(tf.float32, [None, s_dim], name="observation")

			with tf.variable_scope("actor", reuse=reuse):
				#fc1: (mb_size, 64)
				h = ops.fc(self.ob_ph, 64, name="a_fc1")
				h = tf.nn.tanh(h)
				
				#fc2: (mb_size, 64)
				h = ops.fc(h, 64, name="a_fc2")
				h = tf.nn.tanh(h)

				#fc_mean (mb_size, a_dim)
				logits = ops.fc(h, a_dim, name="a_fc_logits")

			with tf.variable_scope("critic", reuse=reuse):
				#fc1: (mb_size, 64)
				h = ops.fc(self.ob_ph, 64, name="c_fc1")
				h = tf.nn.tanh(h)
				
				#fc2: (mb_size, 64)
				h = ops.fc(h, 64, name="c_fc2")
				h = tf.nn.tanh(h)

				#value: (mb_size, 1)
				value = ops.fc(h, 1, name="c_fc_value")

		#value:       (mb_size)
		#action:      (mb_size)
		#neg_logprob: (mb_size)
		self.value = value[:, 0]
		self.distrib = distribs.CategoricalDistrib(logits)
		self.action = self.distrib.sample()
		self.neg_logprob = self.distrib.neg_logp(self.action)


	#---------------------------
	# Forward step
	#---------------------------
	def step(self, mb_obs):
		a, v, nlp = self.sess.run([self.action, self.value, self.neg_logprob], {self.ob_ph: mb_obs})
		return a, v, nlp


	#---------------------------
	# Forward step for 
	# value function
	#---------------------------
	def value_step(self, mb_obs):
		return self.sess.run(self.value, {self.ob_ph: mb_obs})


	#---------------------------
	# Forward step for 
	# sampling actions
	#---------------------------
	def action_step(self, mb_obs):
		return self.sess.run(self.action, {self.ob_ph: mb_obs})